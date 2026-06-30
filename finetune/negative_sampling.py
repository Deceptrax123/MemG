import torch
import numpy as np
from tqdm import tqdm
import multiprocessing
import json
import os
from datasets import load_from_disk
import glob
import bm25s
import traceback
import math

DATA_DIR = "../data_binaries/marco_query_generated/msmarco_flan_t5_xl_queries_generated"
OUTPUT_DIR = "../data_binaries"
NUM_WORKERS = 9
BUFFER_SIZE = 5
TOP_K_NEGATIVES = 20


def get_all_chunks(data_dir):
    search_path = os.path.join(data_dir, "chunk_*")
    return sorted(glob.glob(search_path))


def worker_fn(worker_id, assigned_chunks, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"negatives_part_{worker_id}.jsonl")

    total_batches = math.ceil(len(assigned_chunks) / BUFFER_SIZE)

    with open(output_file, 'w', encoding='utf-8') as f_out:

        for i in range(0, len(assigned_chunks), BUFFER_SIZE):
            current_batch_paths = assigned_chunks[i: i + BUFFER_SIZE]

            batch_queries = []
            batch_passages = []
            batch_ids = []

            for path in current_batch_paths:
                try:
                    ds = load_from_disk(path)
                    for row in ds:
                        doc_id = str(row.get('query_id'))
                        passage = str(row.get('passage'))
                        query = row.get('generated_query')

                        if query is not None:
                            query = str(query)

                        if doc_id and passage and query:
                            batch_ids.append(doc_id)
                            batch_passages.append(passage)
                            batch_queries.append(query)
                except Exception as e:
                    print(f"[Worker {worker_id}] Error reading {path}: {e}")

            if not batch_queries:
                continue

            try:
                retriever = bm25s.BM25(corpus=batch_passages)
                corpus_tokens = bm25s.tokenize(batch_passages)
                retriever.index(corpus_tokens)

                query_tokens = bm25s.tokenize(batch_queries)
                results_obj = retriever.retrieve(
                    query_tokens, k=TOP_K_NEGATIVES + 1)

                if hasattr(results_obj, 'documents'):
                    retrieved_docs_batch = results_obj.documents
                else:
                    retrieved_docs_batch = results_obj[0]

                count = 0

                for q_idx, row_docs in enumerate(retrieved_docs_batch):

                    positive_text = batch_passages[q_idx]

                    negatives = []

                    for cand_text in row_docs:
                        text = str(cand_text)

                        if text == positive_text:
                            continue

                        negatives.append(text)
                        if len(negatives) >= TOP_K_NEGATIVES:
                            break

                    if negatives:
                        record = {
                            "query": batch_queries[q_idx],
                            "positive": positive_text,
                            "negatives": negatives
                        }
                        f_out.write(json.dumps(record) + "\n")
                        count += 1

                print(
                    f"[Worker {worker_id}] Batch {i//BUFFER_SIZE + 1}/{total_batches} done. Saved {count} triplets.")

            except Exception as e:
                print(f"[Worker {worker_id}] Error: {e}")
                traceback.print_exc()


def main():
    all_chunks = get_all_chunks(DATA_DIR)
    total_chunks = len(all_chunks)
    print(
        f"Main: Found {total_chunks} chunks. Distributing to {NUM_WORKERS} workers.")

    chunk_lists = [[] for _ in range(NUM_WORKERS)]
    for i, chunk in enumerate(all_chunks):
        chunk_lists[i % NUM_WORKERS].append(chunk)

    processes = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=worker_fn,
            args=(i, chunk_lists[i], OUTPUT_DIR)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\n All workers finished.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
