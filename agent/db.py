import faiss
import os
import pickle
import re
from pathlib import Path
from agent.stiefel_node import StiefelGaussianNode
from typing import List
from agent.schemas import *
import dateutil.parser
import torch.nn.functional as F
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
import numpy as np
from agent.openai_config import build_openai_client, openai_model_from_args
from dotenv import load_dotenv
from agent.visualize_planning import PlanningVisualizer
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

        self.llm_client = build_openai_client(self.args)
        self.gen_model_name = openai_model_from_args(self.args)
        self.visualization_count = 0
        self.reranking_visualization_count = 0
        self.retrieval_visualization_count = 0
        self.last_retrieval_plan = None
        self.visualization_conversation_id = None
        self.visualization_question_number = None
        self.visualization_question_type = None

        self.clear()

    def set_visualization_context(
        self, conversation_id, question_number, question_type=None
    ):
        """Route every diagnostic for one question into the same stable folder."""
        self.visualization_conversation_id = str(conversation_id)
        self.visualization_question_number = int(question_number)
        self.visualization_question_type = (
            None if question_type is None else str(question_type)
        )

    @staticmethod
    def _safe_folder_name(value):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
        return safe or "unknown"

    def _visualization_output_dir(self):
        base = Path(getattr(
            self.args, "visualization_dir", "visualizations/locomo_planning"
        ))
        conversation = self._safe_folder_name(
            self.visualization_conversation_id or "unscoped"
        )
        number = (
            f"question_{self.visualization_question_number:03d}"
            if self.visualization_question_number is not None
            else "question_000"
        )
        return base / conversation / number

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

    def visualize_reranking(self, question, records):
        if not getattr(self.args, "visualize_planning", False):
            return None
        if self.reranking_visualization_count >= getattr(
            self.args, "visualization_max_plans", 50
        ):
            return None
        try:
            from agent.visualize_planning import PlanningVisualizer
            output = PlanningVisualizer(
                self._visualization_output_dir(),
                dpi=getattr(self.args, "visualization_dpi", 150),
            ).render_reranking(
                question=question,
                records=records,
                max_candidates=getattr(self.args, "visualization_max_candidates", 30),
            )
            self.reranking_visualization_count += 1
            print(f"  [Visualization] Wrote reranking geometry to {output}")
            return output
        except Exception as exc:
            print(f"  [Visualization] Reranking plot failed: {exc}")
            return None

    def visualize_retrieval_hits(self, question, query_labels, records):
        if not getattr(self.args, "visualize_planning", False):
            return None
        if self.retrieval_visualization_count >= getattr(
            self.args, "visualization_max_plans", 50
        ):
            return None
        try:
            output = PlanningVisualizer(
                self._visualization_output_dir(),
                dpi=getattr(self.args, "visualization_dpi", 150),
            ).render_retrieval_hits(
                question=question,
                query_labels=query_labels,
                records=records,
                max_candidates=getattr(
                    self.args, "visualization_max_candidates", 30
                ),
            )
            self.retrieval_visualization_count += 1
            print(f"  [Visualization] Wrote FAISS retrieval hits to {output}")
            return output
        except Exception as exc:
            print(f"  [Visualization] FAISS retrieval plot failed: {exc}")
            return None

    def visualize_query_answer(self, question, reference_answer, predicted_answer):
        if not getattr(self.args, "visualize_planning", False):
            return None
        if self.visualization_count >= getattr(self.args, "visualization_max_plans", 50):
            return None
        try:
            plan = self.last_retrieval_plan or {}
            labels = ["raw query"]
            texts = [question]
            dense_query = str(plan.get("dense_semantic_query", "")).strip()
            if dense_query and dense_query.casefold() != question.strip().casefold():
                labels.append("dense query")
                texts.append(dense_query)
            seen_queries = {text.strip().casefold() for text in texts}
            for index, hop in enumerate(plan.get("decomposed_queries", []) or [], 1):
                hop = str(hop).strip()
                if hop and hop.casefold() not in seen_queries:
                    labels.append(f"hop {index}")
                    texts.append(hop)
                    seen_queries.add(hop.casefold())
            for index, entity in enumerate(plan.get("target_entities", []) or [], 1):
                entity = str(entity).strip()
                if entity:
                    labels.append(f"target entity {index}: {entity}")
                    texts.append(entity)
            labels.extend(["reference answer", "final answer"])
            texts.extend([str(reference_answer), str(predicted_answer)])
            tokenized = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.args.device)
            with torch.no_grad():
                means, eigenvalues, frames, diagonals, _ = self.encoder.gaussian_projection(tokenized)
            if means.dim() == 1:
                means = means.unsqueeze(0)
                eigenvalues = eigenvalues.unsqueeze(0)
                frames = frames.unsqueeze(0)
                diagonals = diagonals.unsqueeze(0)
            output = PlanningVisualizer(
                self._visualization_output_dir(),
                dpi=getattr(self.args, "visualization_dpi", 150),
            ).render_query_answer(
                question=question,
                reference_answer=str(reference_answer),
                predicted_answer=str(predicted_answer),
                means=means, lambdas=eigenvalues, frames=frames, diagonals=diagonals,
                plan=self.last_retrieval_plan,
                labels=labels,
                texts=texts,
            )
            self.visualization_count += 1
            print(f"  [Visualization] Wrote query-answer geometry to {output}")
            return output
        except Exception as exc:
            print(f"  [Visualization] Query-answer plot failed: {exc}")
            return None

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
                model=self.gen_model_name,
                response_model=RetrievalPlan,
                messages=[{"role": "system", "content": "You are a precise database query planner."},
                          {"role": "user", "content": prompt}]
            )
        except Exception as e:
            print(
                f"  [Plan] LLM Planning failed: {e}. Falling back to raw query.")
            plan = RetrievalPlan(
                difficulty_level="single_hop", dense_semantic_query=query_text,
                decomposed_queries=[query_text], target_entities=[],
                search_keywords=[], temporal_intent="all"
            )

        self.last_retrieval_plan = (
            plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
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
                lambdas = lambdas.unsqueeze(0)
                Vs = Vs.unsqueeze(0)
                ds = ds.unsqueeze(0)

            d_vecs = torch.diagonal(ds, dim1=-2, dim2=-1)
            sigma_effs = d_vecs + torch.sum(Vs.square() * lambdas.unsqueeze(1), dim=-1)

        query_node = StiefelGaussianNode(
            node_id="q_0", text=plan.dense_semantic_query, topic="query",
            entities=plan.target_entities, timestamp=float('inf'),
            mu_raw=mus[0], V_raw=Vs[0], d_raw=ds[0], topic_embed=None
        ).to(self.args.device)

        retrieved_nids = set()
        faiss_hit_records = []

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

                    start_idx = 1
                    num_intersected = 0

                    for q_idx in range(start_idx, num_queries):
                        mu_k = mus[q_idx].squeeze()
                        var_k = sigma_effs[q_idx].squeeze()

                        sum_inv_var += 1.0 / (var_k + eps)
                        sum_mean_over_var += mu_k / (var_k + eps)
                        num_intersected += 1

                    var_int = 1.0 / (sum_inv_var + eps)
                    mu_int = var_int * sum_mean_over_var

                    q_vec_int = torch.cat(
                        [mu_int, torch.sqrt(var_int)], dim=-1)

                    q_vecs_list.append(q_vec_int)

                q_vecs = torch.stack(
                    q_vecs_list).cpu().numpy().astype('float32')

            dense_k = self.index.ntotal
            distances_matrix, indices_matrix = self.index.search(q_vecs, dense_k)

            query_labels = ["dense query"] + [
                f"hop {index}" for index in range(1, len(queries_to_embed))
            ]
            if len(q_vecs_list) > len(query_labels):
                query_labels.append("multi-hop intersection")
            capture_k = (
                min(dense_k, getattr(self.args, "visualization_max_candidates", 30))
                if getattr(self.args, "visualize_planning", False)
                else 0
            )
            for query_index, (distances, indices) in enumerate(
                zip(distances_matrix, indices_matrix)
            ):
                query_vector = q_vecs[query_index]
                for hit_rank, (distance, idx) in enumerate(
                    zip(distances[:capture_k], indices[:capture_k]), 1
                ):
                    idx = int(idx)
                    if idx == -1 or idx not in self.faiss_id_map:
                        continue
                    node_id = self.faiss_id_map[idx]
                    candidate_vector = self.index.reconstruct(idx)
                    mean_energy = float(np.sum(
                        (query_vector[:self.dim] - candidate_vector[:self.dim]) ** 2
                    ))
                    std_profile_energy = float(np.sum(
                        (query_vector[self.dim:] - candidate_vector[self.dim:]) ** 2
                    ))
                    node = self.nodes_dict[node_id]
                    faiss_hit_records.append({
                        "query_index": query_index,
                        "query_label": query_labels[query_index],
                        "query_text": (
                            queries_to_embed[query_index]
                            if query_index < len(queries_to_embed)
                            else "product-of-Gaussians multi-hop intersection"
                        ),
                        "hit_rank": hit_rank,
                        "faiss_index": idx,
                        "node_id": node_id,
                        "text": node.text,
                        "timestamp": node.timestamp,
                        "topic": node.topic,
                        "mean_energy": mean_energy,
                        "std_profile_energy": std_profile_energy,
                        "wasserstein_squared": float(distance),
                        "component_sum": mean_energy + std_profile_energy,
                    })

            for indices in indices_matrix:
                for idx in indices:
                    if idx != -1 and idx in self.faiss_id_map:
                        retrieved_nids.add(self.faiss_id_map[idx])

            if faiss_hit_records:
                self.visualize_retrieval_hits(
                    query_text, query_labels, faiss_hit_records
                )

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
