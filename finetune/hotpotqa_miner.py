import json
from datasets import load_dataset
from tqdm import tqdm
import os


class HotpotFoldMiner:
    def __init__(self, num_folds=4):
        print("Loading HotpotQA (Distractor Setting)...")
        self.dataset = load_dataset("hotpot_qa", "distractor")
        self.num_folds = num_folds

    def mine(self, output_dir="training_data"):
        os.makedirs(output_dir, exist_ok=True)

        files = []
        for i in range(self.num_folds):
            f_path = os.path.join(output_dir, f"hotpot_fold_{i}.jsonl")
            files.append(open(f_path, 'w', encoding='utf-8'))
            print(f"Created fold file: {f_path}")

        print("Processing...")

        for idx, row in enumerate(tqdm(self.dataset['train'])):

            question = row['question']

            pos_titles = row['supporting_facts']['title']
            pos_sent_ids = row['supporting_facts']['sent_id']
            gold_set = set(zip(pos_titles, pos_sent_ids))

            positives = []
            negatives = []

            titles = row['context']['title']
            all_sentences_lists = row['context']['sentences']

            for i, title in enumerate(titles):
                sentences = all_sentences_lists[i]

                for sent_idx, sentence in enumerate(sentences):
                    if len(sentence) < 5:
                        continue

                    if (title, sent_idx) in gold_set:
                        positives.append(sentence)
                    else:
                        negatives.append(sentence)

            if not positives or not negatives:
                continue

            fold_id = idx % self.num_folds

            for pos_text in positives:
                record = {
                    "source": "hotpot_qa",
                    "id": f"{row['id']}_{fold_id}",
                    "query": question,
                    "positive": pos_text,
                    "negatives": negatives[:50]
                }

                files[fold_id].write(json.dumps(record) + "\n")

        for f in files:
            f.close()
        print("Done!")


if __name__ == "__main__":
    miner = HotpotFoldMiner(num_folds=4)
    miner.mine("training_data/hotpot")
