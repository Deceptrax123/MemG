import os
import torch
import networkx as nx
import faiss
from agent.schemas import *
import concurrent.futures
from dateutil import parser as date_parser
from agent.optimizer import RiemannianCayleySGD
from sentence_transformers import SentenceTransformer
import time
from agent.db import LocalGeometricDB
from agent.policy import AGMRiemmannianPolicy
from agent.math_utils import *
from agent.openai_config import build_openai_client, openai_model_from_args


class GeoLogicMemorySystem:
    def __init__(self, args, trained_encoder_model, tokenizer):
        args.device = 'cuda' if args.cuda is True else 'cpu'
        self.args = args
        self.encoder = trained_encoder_model.to(args.device).eval()
        self.tokenizer = tokenizer
        self.dim = args.output_dim
        self.db = LocalGeometricDB(
            args, encoder=self.encoder, tokenizer=self.tokenizer)

        self.llm_client = build_openai_client(self.args)
        self.gen_model_name = openai_model_from_args(self.args)

        self.nodes_dict = {}

        self.index = faiss.IndexFlatIP(self.dim * 2)
        self.id_to_key = {}
        self.counter = 0
        self.policy = AGMRiemmannianPolicy(
            self.args.alpha_success, self.args.beta_consistency, self.args.gamma_inertia)
        self.embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2').to(args.device)

    def _extract_single_window(self, i, window):
        dialogue_text = "\n".join(
            [f"[{t['date_time']}] {t['speaker']}: {t['text']}" for t in window])

        fallback_ts = date_parser.parse(
            window[-1]['date_time']).timestamp()

        prompt = f"""You are a precise memory encoder for a long-term memory system. 
                Your task is to extract EVERY piece of factual information from the dialogue and convert it into structured memory entries.

                CURRENT WINDOW START TIME is given in the dialogue window text.

                [Requirements]
                1. Information Filtering:
                - Discard social filler, acknowledgements, and conversational routines that introduce
                no new factual or semantic information.
                - Discard redundant confirmations unless they modify or finalize a decision
                
                2. Complete Coverage: Generate enough memory entries to ensure ALL information in the dialogues is captured.
                
                3. Strict Speaker Disambiguation (CRITICAL): The dialogue lines start with the speaker's name. When a speaker uses "I", "my", or "mine", you MUST replace it with their exact speaker name. 
                - Bad: "The User likes camping."
                - Good: "Alice likes camping."
                - Good: "Bob hates camping."
                
                4. Force Disambiguation: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that). Replace them with the specific entity they refer to.
                
                5. Lossless Information: Each entry's lossless_restatement must be a complete, independent, understandable sentence. 
                
                6. Precise Extraction:
                - keywords: Core keywords (names, places, entities, topic words)
                - timestamp: Absolute time in ISO 8601 format (if explicit time mentioned)
                - location: Specific location name
                - persons: ALL exact speaker names and person names mentioned
                - entities: Companies, products, organizations, etc.
                - topic: The topic of this information

                [Output Format]
                Return a JSON array, each element is a memory entry:
                {{
                "memory_entries": [
                    {{
                    "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)",
                    "keywords": ["keyword1", "keyword2"],
                    "timestamp": "YYYY-MM-DDTHH:MM:SS", 
                    "location": "location name",
                    "persons": ["name1", "name2"],
                    "entities": ["entity1", "entity2"],
                    "topic": "topic phrase"
                    }}
                ]
                }}

                [Example]
                Dialogues:
                [2025-11-15T14:30:00] Alice: Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product
                [2025-11-15T14:31:00] Bob: Okay, I'll prepare the materials

                Output:
                {{
                "memory_entries": [
                    {{
                    "lossless_restatement": "Alice suggested at 2025-11-15T14:30:00 to meet with Bob at Starbucks on 2025-11-16T14:00:00 to discuss the new product.",
                    "keywords": ["Alice", "Bob", "Starbucks", "new product", "meeting"],
                    "timestamp": "2025-11-16T14:00:00",
                    "location": "Starbucks",
                    "persons": ["Alice", "Bob"],
                    "entities": ["new product"],
                    "topic": "Product discussion meeting arrangement"
                    }},
                    {{
                    "lossless_restatement": "Bob agreed to attend the meeting with Alice and committed to prepare relevant materials.",
                    "keywords": ["Bob", "prepare materials", "agree"],
                    "timestamp": null,
                    "location": null,
                    "persons": ["Bob", "Alice"],
                    "entities": [],
                    "topic": "Meeting preparation confirmation"
                    }}
                ]
                }}

                [Input Dialogues]
                {dialogue_text}
                """

        try:
            extraction = self.llm_client.chat.completions.create(model=self.gen_model_name, response_model=MemoryExtractionList, messages=[
                {'role': 'user', 'content': prompt}], temperature=0.1)

            extracted_facts = []
            for entry in extraction.memory_entries:
                combined_entities = set(
                    entry.keywords + entry.persons + entry.entities)
                if entry.location:
                    combined_entities.add(entry.location)

                fact_time = entry.timestamp if entry.timestamp else fallback_ts

                extracted_facts.append(
                    AtomicFact(
                        fact=entry.lossless_restatement,
                        topic=entry.topic,
                        entities=list(combined_entities),
                        timestamp=fact_time
                    )
                )

            return i, extracted_facts
        except Exception as e:
            print(f"    [Worker] Window {i} extraction failed: {e}")
            return i, []

    def process_windows_parallel(self, turns, window_size, overlap_size, max_workers):
        step_size = max(1, window_size-overlap_size)

        windows_to_process = []
        for i in range(0, len(turns), step_size):
            window = turns[i:i+window_size]

            if window:
                windows_to_process.append((i, window))
        if not windows_to_process:
            return

        print(
            f"  [Parallel Ingestion] Dispatching {len(windows_to_process)} windows to {max_workers} workers...")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_window = {
                executor.submit(self._extract_single_window, idx, w): idx for idx, w in windows_to_process
            }

            for future in concurrent.futures.as_completed(future_to_window):
                idx = future_to_window[future]

                try:
                    res_idx, facts = future.result()
                    results.append((res_idx, facts))
                except Exception as e:
                    print(
                        f"[Pool] Error retrieving result for window {idx}: {e}")
        results.sort(key=lambda x: x[0])

        total_facts = 0
        for _, facts in results:
            if facts:
                self.db.ingest_facts(facts)
                total_facts += len(facts)
        print(
            f"  [Parallel Ingestion] Complete. Added {total_facts} total facts to Vector Store.")

    def retrieve_and_revise(self, query_text, top_k, run_optimization=False):

        query_node, retrieved_nodes = self.db.retrieve_context_bag(query_text)

        if not retrieved_nodes:
            return []

        # with torch.no_grad():
        #     edges = FastGeometricResolver.build_and_resolve_graph(
        #         retrieved_nodes, overlap_threshold=0.6, topic_sim_threshold=0.75)

        # if run_optimization:
        #     print(
        #         f"  [Agent] Running heavy 30-step Riemannian geometric optimization...")
        #     policy = AGMRiemmannianPolicy()
        #     retrieved_nodes = policy.execute_rollout(
        #         query_node, retrieved_nodes, edges)

        ranked_nodes = []
        q_mu, q_V, _, q_d = query_node.get_params()

        with torch.no_grad():
            for node in retrieved_nodes:
                n_mu, n_V, _, n_d = node.get_params()

                dist = GaussianMath.geometric_energy(
                    q_mu, q_V, q_d, n_mu, n_V, n_d
                ).item()

                ranked_nodes.append((dist, node))

        ranked_nodes.sort(key=lambda x: x[0])

        final_nodes = [node for dist, node in ranked_nodes[:40]]
        final_nodes.sort(key=lambda n: n.timestamp)

        retrived_top_nodes = GeometricRetriever.get_top_k_geometric(
            query_node, final_nodes, k=max(top_k, top_k))

        verified_facts = []
        for node in retrived_top_nodes:
            verified_facts.append({
                "text": node.text,
                "timestamp": node.timestamp,
                # "status": getattr(node, "status", "UNCERTAIN"),
                "topic": getattr(node, "topic", "Unknown"),
                "entities": getattr(node, "entities", [])
            })

        return verified_facts

    def answer_question(self, question, top_k):
        verified_facts = self.retrieve_and_revise(
            question, top_k=top_k, run_optimization=False)

        if not verified_facts:
            print("  [Agent] No consistent memories found.")
            return "I do not have enough information in my memory.", "Memory contained no facts that overlap with query."

        print(
            f"  [Agent] Riemannian optimization complete. Retained {len(verified_facts)} verified facts.")

        verified_facts.sort(key=lambda x: x.get('timestamp', 0.0))

        unique_topics = set()
        unique_entities = set()

        for fact in verified_facts:
            if 'topic' in fact and fact['topic']:
                unique_topics.add(fact['topic'])
            if 'entities' in fact and fact['entities']:
                for ent in fact['entities']:
                    unique_entities.add(ent)

        abstract_str = ""
        if unique_topics:
            abstract_str += f"- Recurring Topics/Themes: {', '.join(unique_topics)}\n"
        if unique_entities:
            abstract_str += f"- Interacting Entities/Places: {', '.join(unique_entities)}\n"

        if not abstract_str:
            abstract_str = "No high-level abstract metadata available for these facts."

        context_str = ""
        for fact in verified_facts:
            text = fact.get('text', fact.get('fact', ''))
            status = fact.get('status', 'UNCERTAIN')
            ts = fact.get('timestamp', 'Unknown Time')

            context_str += f"[Time: {ts}] Fact: {text} (Status: {status})\n"

        prompt = f"""You are an assistant with access to a structured long-term memory graph.

                    USER QUERY:
                    {question}

                    RETRIEVED MEMORY (Ordered Chronologically):

                    [ABSTRACT REPRESENTATIONS]:
                    {abstract_str}

                    [DETAILED MEMORY UNITS]:
                    {context_str}

                    INSTRUCTIONS:
                    1. Hierarchical Reasoning:
                    - Use abstract representations to capture recurring patterns or general user
                    preferences.
                    
                    - Use detailed memory units to ground the response with specific facts.

                    2. Detail Preservation (CRITICAL):
                    - Do not assume a recent general reflection cancels out an older specific event. Treat them as complementary facts. (e.g., if an older memory has a specific date and a newer memory just talks generally about the topic, use the specific date from the older memory).

                    3. Faithfulness & Constraints:
                    - Base the answer STRICTLY on the retrieved memory.
                    - IF THE REQUIRED INFORMATION IS COMPLETELY MISSING, RESPOND EXACTLY WITH: "I do not have enough information in my memory."
                    - CONCISE EXTRACTION: Answer the query precisely. If the answer is just an exact entity, name, or date, JUST MENTION THAT ONLY. DO NOT PROVIDE A WHOLE SENTENCE FOR IT.
                    - FOR DATE IF ALL INFORMATION IS PRESENT OR MONTH AND YEAR OR MONTH DAY PRESENT ANSWER AS DATE MONTH IN WORDS AND YEAR. 
                    """
        try:
            response = self.llm_client.chat.completions.create(
                model=self.gen_model_name,
                response_model=FreeTextResponse,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict logical interpreter of a geometric memory graph. You deduce truth based on abstract patterns, mathematical variance, and temporal progression."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0
            )

            return response.answer, response.rationale

        except Exception as e:
            print(f"  [Agent] LLM Generation Error: {e}")
            return "I do not have enough information in my memory.", str(e)
