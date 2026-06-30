import torch
from tqdm import tqdm
from utils import *
from datasets import Dataset as ds
from torch.utils.data import Dataset, DataLoader


class PassageDataset(Dataset):
    def __init__(self, marco_ds):
        self.ds = marco_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return {
            "query_id": self.ds[idx]["query_id"],
            "passage": self.ds[idx]["passage"],
        }


class QueryPassage:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.save_every = 500
        self.out_dir = (
            "../data_binaries/marco_query_generated/"
            "msmarco_flan_t5_xl_queries_generated"
        )

        print("Loading model and tokenizer...")
        self.tokenizer, self.model = load_query_model(
            self.device, self.args.gpu_id)

        if self.device == "cuda":
            torch.cuda.set_device(args.gpu_id)

        self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)
        print("Model loaded")

        print("Loading MS MARCO passages...")
        raw_ds = load_passages()

        print("Flattening positive passages (generator-based)...")
        self.passage_dataset = ds.from_generator(
            lambda: positive_passage_generator(raw_ds)
        )

        self.passage_dataset = self.passage_dataset.remove_columns(
            [c for c in self.passage_dataset.column_names
             if c not in ("query_id", "passage")]
        )

        final_dataset = PassageDataset(self.passage_dataset)

        loader = DataLoader(
            final_dataset,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        print("Generating queries...")
        self._run_generation(loader)

    def generate_query(self, passages, max_new_tokens=16):
        prompts = [build_prompt(p) for p in passages]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
        )

        return self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

    def _run_generation(self, loader):
        results = []
        chunk_id = 0

        SAVE_EVERY_BATCHES = 1000

        for batch_idx, batch in enumerate(tqdm(loader), start=1):
            qids = batch["query_id"]
            passages = batch["passage"]

            queries = self.generate_query(passages)

            for qid, p, q in zip(qids, passages, queries):
                results.append({
                    "query_id": qid,
                    "passage": p,
                    "generated_query": q
                })

            if batch_idx <= 2 or batch_idx % 100 == 0:
                print("\n--- SAMPLE ---")
                print("PASSAGE:", passages[0][:200], "...")
                print("QUERY  :", queries[0])
                print("-------------")

            if batch_idx % SAVE_EVERY_BATCHES == 0:
                self._save_chunk(results, chunk_id)
                results.clear()
                chunk_id += 1
                torch.cuda.empty_cache()

        if results:
            self._save_chunk(results, chunk_id)

    print("Query generation completed")

    def _save_chunk(self, results, chunk_id):
        print(f"\nSaving chunk {chunk_id} ({len(results)} examples)")

        chunk_ds = ds.from_list(results)
        chunk_path = f"{self.out_dir}/chunk_{chunk_id:05d}"

        chunk_ds.save_to_disk(chunk_path)

        print(f"Saved to {chunk_path}")
