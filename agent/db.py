import instructor
import faiss
import os
import pickle
from stiefel_node import StiefelGaussianNode
from typing import List
from schemas import *
import dateutil.parser
import torch.nn.functional as F
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
import numpy as np
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv('.env')


class LocalGeometricDB:
    def __init__(self, args, encoder, tokenizer, cache_dir="./memory_cache"):
        self.args = args
        self.tokenizer = tokenizer
        self.encoder = encoder.to(args.device).eval()
        self.dim = self.args.output_dim
        self.cache_dir = cache_dir
        self.embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2').to(args.device)
        os.makedirs(cache_dir, exist_ok=True)

        if self.args.provider == 'azure':
            client = AzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_KEY'),
                api_version="2024-12-01-preview",
                azure_endpoint=self.args.azure_endpoint
            )

            self.llm_client = instructor.from_openai(
                client)
        elif self.args.provider == 'open_router':
            client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url='https://openrouter.ai/api/v1'
            )
            self.llm_client = instructor.from_openai(client)
        else:
            self.llm_client = instructor.from_provider(
                'openai/gpt-4.1-mini'
            )

        self.clear()

    def clear(self):
        self.nodes_dict = {}

        self.corpus_tokens = []
        self.bm25 = None

        self.index = faiss.IndexFlatL2(self.dim * 2)
        self.faiss_id_map = {}
        self.counter = 0

    def save_session(self, session_id: str):
        path = os.path.join(self.cache_dir, f"{session_id}.pkl")

        nodes_data = {nid: {
            'text': n.text, 'timestamp': n.timestamp, 'topic': n.topic, 'entities': n.entities, 'topic_embedding': n.topic_embedding,
            'state_dict': n.state_dict()
        } for nid, n in self.nodes_dict.items()}

        faiss_vecs = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * self.index.d).reshape(
            self.index.ntotal, self.index.d) if self.index.ntotal > 0 else None

        state = {
            'nodes_data': nodes_data,
            'corpus_tokens': self.corpus_tokens,
            'faiss_id_map': self.faiss_id_map,
            'counter': self.counter,
            'faiss_vectors': faiss_vecs
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_session(self, session_id: str) -> bool:
        path = os.path.join(self.cache_dir, f"{session_id}.pkl")
        if not os.path.exists(path):
            return False

        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.faiss_id_map = state['faiss_id_map']
        self.corpus_tokens = state.get('corpus_tokens', [])
        self.counter = state['counter']

        self.nodes_dict = {}
        for nid, d in state['nodes_data'].items():
            node = StiefelGaussianNode(nid, d['text'], d['topic'], d['entities'], d['timestamp'],
                                       torch.zeros(self.dim), torch.eye(self.dim)[:, :self.args.rank], torch.ones(self.dim), d['topic_embedding']).to(self.args.device)
            node.load_state_dict(d['state_dict'])
            self.nodes_dict[nid] = node

        self.index = faiss.IndexFlatL2(self.dim * 2)
        if state['faiss_vectors'] is not None:
            self.index.add(state['faiss_vectors'])

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

        return True

    def ingest_facts(self, extracted_facts: List['AtomicFact']):
        if not extracted_facts:
            return

        novel_facts = []

        for f in extracted_facts:
            is_duplicate = False
            for nid, node in self.nodes_dict.items():
                if fuzz.token_sort_ratio(node.text.lower(), f.fact.lower()) > 95.0:
                    fact_time = f.timestamp
                    if isinstance(fact_time, str):
                        try:
                            fact_time = dateutil.parser.isoparse(
                                fact_time).timestamp()
                        except ValueError:
                            fact_time = 0.0

                    node.timestamp = max(node.timestamp, fact_time)
                    is_duplicate = True
                    break

            if not is_duplicate:
                novel_facts.append(f)

        if not novel_facts:
            return

        texts = [f.fact for f in novel_facts]
        texts_tokenized = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        ).to(self.args.device)

        with torch.no_grad():
            mus, lambdas, Vs, ds, _ = self.encoder.gaussian_projection(
                texts_tokenized)

        for j, f in enumerate(novel_facts):
            nid = f"mem_{self.counter}"

            fact_time = f.timestamp
            if isinstance(fact_time, str):
                try:
                    fact_time = dateutil.parser.isoparse(fact_time).timestamp()
                except ValueError:
                    fact_time = 0.0

            raw_embed = self.embedding_model.encode(
                f.topic, convert_to_tensor=True)
            normalized_emb = F.normalize(
                raw_embed.unsqueeze(0), p=2, dim=1).squeeze(0)

            node = StiefelGaussianNode(
                node_id=nid, text=f.fact, topic=f.topic, entities=f.entities,
                timestamp=fact_time, mu_raw=mus[j], V_raw=Vs[j], d_raw=ds[j], topic_embed=normalized_emb
            ).to(self.args.device)

            self.nodes_dict[nid] = node

            doc_string = f"{f.fact} {f.topic} {' '.join(f.entities)}"
            tokens = doc_string.lower().split()
            self.corpus_tokens.append(tokens)

            with torch.no_grad():
                d_vec = torch.diagonal(ds[j])

                v_squared = Vs[j]**2
                v_lambda_squared = v_squared*lambdas[j]
                v_diag = torch.sum(v_lambda_squared, dim=-1)
                sigma_eff = d_vec + v_diag

                flat_tensor = torch.cat(
                    [mus[j], torch.sqrt(sigma_eff)], dim=-1)
                flat_vec = flat_tensor.cpu().numpy().astype('float32')
                flat_vec_2d = flat_vec.reshape(1, -1)

            self.index.add(flat_vec_2d)
            self.faiss_id_map[self.counter] = nid
            self.counter += 1

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve_context_bag(self, query_text: str):
        print(f"\n  [Retrieval] Planning search for: '{query_text}'")

        prompt = f"""You are an advanced geometric database query planner.
                    Analyze the user's question, determine its exact logical complexity, and formulate a precise retrieval strategy.

                    USER QUESTION:
                    {query_text}

                    INSTRUCTIONS:
                    1. 'dense_semantic_query': Rewrite the user's question into a broad, fully self-contained declarative search string. Pay absolute attention to WHO is being asked about. If the question asks about a specific speaker (e.g., "Alice" or "Bob"), ensure that exact name is the clear subject of the query. NEVER use generic terms like "the user" or "the speaker"—use their actual name.

                    2. 'difficulty_level': Classify strictly as 'single_hop' (one direct fact), 'multi_hop' (requires bridging multiple facts), or 'open_domain' (broad summary).

                    3. 'decomposed_queries': THIS IS THE MOST CRITICAL STEP FOR MULTI-HOP REASONING.
                    - FOR MULTI-HOP ONLY: Break the query into exactly 2 or 3 independent, declarative sub-queries representing the missing logical links.
                    - CRITICAL RULE: Every sub-query MUST be fully self-contained. NEVER use pronouns (his, her, their, it). NEVER use "the user"—use the actual name of the person.
                    - Good Example ("What is my manager's dog's name?" - asked about Alice):
                        ["The identity of Alice's manager.",
                            "The name of the dog belonging to Alice's manager."]
                    - Good Example ("Where did my brother work before moving to Seattle?" - asked about John):
                        ["The identity of John's brother.", "The date John's brother moved to Seattle.",
                            "The workplace of John's brother before moving to Seattle."]
                    - FOR SINGLE-HOP / OPEN-DOMAIN: Leave this list EMPTY.

                    4. 'target_entities': Extract exact proper nouns (names, cities, companies). You MUST include the specific person's name (e.g., "Alice", "John") if the question asks about their specific perspective, relatives, or actions! Do not guess or hallucinate unmentioned entities.

                    5. 'search_keywords': Extract 2-4 crucial specific objects, action verbs, or EXPLICIT DATES/YEARS (e.g., "blog", "camping", "bought", "June", "2023"). THIS IS CRITICAL FOR TEMPORAL AND KEYWORD SEARCH. Do not include generic words.

                    6. 'temporal_intent':
                    - 'current': Asking for the latest, up-to-date state.
                    - 'historical': Asking for a specific past state (e.g., "Where did Bob used to live?").
                    - 'trajectory': Asking for a timeline or comparison (e.g., "Where did Sarah work before Apple?").
                    - 'all': General facts with no specific time constraints.
                    """
        try:
            plan = self.llm_client.chat.completions.create(
                model="openai/gpt-4.1-mini",
                response_model=RetrievalPlan,
                messages=[{"role": "system", "content": "You are a precise database query planner."},
                          {"role": "user", "content": prompt}]
            )
        except Exception as e:
            print(
                f"  [Plan] LLM Planning failed: {e}. Falling back to raw query.")
            plan = RetrievalPlan(
                difficulty_level="single_hop", dense_semantic_query=query_text,
                decomposed_queries=[
                    query_text], target_entities=[], temporal_intent="all"
            )

        queries_to_embed = [plan.dense_semantic_query]
        if plan.decomposed_queries:
            queries_to_embed.extend(plan.decomposed_queries)

        search_tokenized = self.tokenizer(
            queries_to_embed, padding=True, truncation=True, return_tensors='pt'
        ).to(self.args.device)

        with torch.no_grad():
            mus, lambdas, Vs, ds, _ = self.encoder.gaussian_projection(
                search_tokenized)

            if mus.dim() == 1:
                mus = mus.unsqueeze(0)
                Vs = Vs.unsqueeze(0)
                ds = ds.unsqueeze(0)

            d_vecs = torch.diagonal(ds, dim1=-2, dim2=-1)
            v_squared = Vs**2
            v_lambda_squared = v_squared*lambdas
            sigma_effs = d_vecs + torch.sum(v_lambda_squared, dim=-1)

        query_node = StiefelGaussianNode(
            node_id="q_0", text=plan.dense_semantic_query, topic="query",
            entities=plan.target_entities, timestamp=float('inf'),
            mu_raw=mus[0], V_raw=Vs[0], d_raw=ds[0], topic_embed=None
        ).to(self.args.device)

        retrieved_nids = set()

        if self.index.ntotal > 0:
            with torch.no_grad():
                q_vecs_list = [
                    torch.cat([mus[0], torch.sqrt(sigma_effs[0])], dim=-1)]

                num_queries = len(queries_to_embed)
                if num_queries > 1:
                    for q_idx in range(1, num_queries):
                        q_vecs_list.append(
                            torch.cat([mus[q_idx], torch.sqrt(sigma_effs[q_idx])], dim=-1))

                if plan.difficulty_level == "multi_hop" and num_queries >= 2:
                    eps = 1e-8

                    sum_inv_var = torch.zeros_like(sigma_effs[1].squeeze())
                    sum_mean_over_var = torch.zeros_like(mus[1].squeeze())

                    start_idx = 1 if num_queries >= 3 else 0
                    num_intersected = 0

                    for q_idx in range(start_idx, num_queries):
                        mu_k = mus[q_idx].squeeze()
                        var_k = sigma_effs[q_idx].squeeze()

                        sum_inv_var += 1.0 / (var_k + eps)
                        sum_mean_over_var += mu_k / (var_k + eps)
                        num_intersected += 1

                    var_int = float(num_intersected) / (sum_inv_var + eps)

                    mu_int = (var_int / float(num_intersected)) * \
                        sum_mean_over_var

                    q_vec_int = torch.cat(
                        [mu_int, torch.sqrt(var_int)], dim=-1)

                    if q_vec_int.dim() == 1:
                        q_vec_int = q_vec_int.unsqueeze(0)

                    q_vecs_list.append(q_vec_int)

                q_vecs = torch.stack(
                    q_vecs_list).cpu().numpy().astype('float32')

            dense_k = self.index.ntotal
            _, indices_matrix = self.index.search(q_vecs, dense_k)

            for indices in indices_matrix:
                for idx in indices:
                    if idx != -1 and idx in self.faiss_id_map:
                        retrieved_nids.add(self.faiss_id_map[idx])

        # if self.bm25:
        #     sparse_query = f"{' '.join(queries_to_embed)} {' '.join(plan.target_entities)} {' '.join(plan.search_keywords)}"
        #     tokenized_query = sparse_query.lower().split()

        #     doc_scores = self.bm25.get_scores(tokenized_query)

        #     sparse_k = 25 if plan.difficulty_level == "open_domain" or plan.difficulty_level == 'multi_hop' else 15
        #     top_bm25_indices = np.argsort(doc_scores)[::-1][:sparse_k]

        #     for idx in top_bm25_indices:
        #         if doc_scores[idx] > 0 and idx in self.faiss_id_map:
        #             retrieved_nids.add(self.faiss_id_map[idx])

        context_nodes = [self.nodes_dict[nid] for nid in retrieved_nids]
        if plan.temporal_intent in ["historical", "trajectory"]:
            context_nodes = sorted(context_nodes, key=lambda x: x.timestamp if isinstance(
                x.timestamp, (int, float)) else 0.0)

        return query_node, context_nodes
