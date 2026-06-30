from datasets import load_dataset
import random
import torch
import numpy as np
import pytz
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_passages():
    ds = load_dataset("microsoft/ms_marco", "v2.1", split='train')
    print(ds[0]['passages']['is_selected'])

    return ds


def print_local_time():
    utc_dt = datetime.now(timezone.utc)
    PST = pytz.timezone('Asia/Kolkata')
    print("Pacific time {}".format(utc_dt.astimezone(PST).isoformat()))
    return


def build_prompt(passage):
    return (
        "Generate ONE concise web search query.\n"
        "Do not ask a question.\n"
        "Do not add punctuation.\n"
        f"Document:\n{passage}\n"
        "Query:"
    )


def positive_passage_generator(ds):
    for ex in ds:
        qid = ex["query_id"]
        passages = ex["passages"]

        if isinstance(passages, dict):
            texts = passages["passage_text"]
            labels = passages["is_selected"]
            for t, l in zip(texts, labels):
                if l == 1:
                    yield {
                        "query_id": qid,
                        "passage": t
                    }

        elif isinstance(passages, list):
            for p in passages:
                if p.get("is_selected", 0) == 1:
                    yield {
                        "query_id": qid,
                        "passage": p["passage_text"]
                    }


def extract_selected_passages(batch):
    out_query_ids = []
    out_passages = []

    query_ids = batch["query_id"]
    passages_col = batch["passages"]

    for idx in range(len(query_ids)):
        qid = query_ids[idx]
        p = passages_col[idx]

        if isinstance(p, dict):
            texts = p["passage_text"]
            labels = p["is_selected"]
            for t, l in zip(texts, labels):
                if l == 1:
                    out_query_ids.append(qid)
                    out_passages.append(t)

        elif isinstance(p, list):
            for item in p:
                if item.get("is_selected", 0) == 1:
                    out_query_ids.append(qid)
                    out_passages.append(item["passage_text"])

        else:
            continue

    assert len(out_query_ids) == len(out_passages)

    return {
        "query_id": out_query_ids,
        "passage": out_passages
    }


def load_query_model(device, gpu_id):
    model_path = '/home/models/flan-t5-xl'
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=f"{device}:{gpu_id}")
    model.eval()

    return tokenizer, model
