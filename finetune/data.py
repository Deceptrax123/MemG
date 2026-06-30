import json
import random
from torch.utils.data import IterableDataset, DataLoader
import itertools
import glob


class MixedStreamerDataset(IterableDataset):
    def __init__(self, args, marco_files, hotpot_files, prob_hotpot=0.3, num_hard_negatives=64):
        self.marco_files = marco_files
        self.hotpot_files = hotpot_files
        self.prob_hotpot = prob_hotpot
        self.args = args
        self.num_hard_negatives = num_hard_negatives

    def parse_file_infinite(self, files):
        for file_path in itertools.cycle(files):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield json.loads(line)

    def __iter__(self):
        marco_iter = self.parse_file_infinite(self.marco_files)
        hotpot_iter = self.parse_file_infinite(self.hotpot_files)

        while True:
            if random.random() < self.prob_hotpot:
                data = next(hotpot_iter)
            else:
                data = next(marco_iter)

            negs = data['negatives']
            if len(negs) > self.num_hard_negatives:
                hard_negs = random.sample(negs, self.num_hard_negatives)
            else:
                hard_negs = negs

            while len(hard_negs) < self.num_hard_negatives:
                hard_negs.append(random.choice(negs))

            if self.args.model_name == 'nomic-embed-text-v1.5':
                prefix_q = 'search_query: '
                prefix_d = 'search_document: '
                yield prefix_q + data['query'], prefix_d + data['positive'], [prefix_d + n for n in hard_negs]
            else:
                yield data['query'], data['positive'], hard_negs


def collate_function(batch):
    queries = [b[0] for b in batch]
    positives = [b[1] for b in batch]
    negatives = [b[2] for b in batch]

    return queries, positives, negatives


def load_data(args):
    MARCO_FILES = glob.glob("../data_binaries/*.jsonl")
    HOTPOT_FILES = glob.glob("../data_binaries/hotpot/*.jsonl")
    dataset = MixedStreamerDataset(args=args,
                                   marco_files=MARCO_FILES, hotpot_files=HOTPOT_FILES)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_function)

    return dataloader
