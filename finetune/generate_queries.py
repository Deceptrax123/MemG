import torch
import argparse
import os
from finetune.utils import print_local_time, set_seed
import wandb
import time
from dataset import QueryPassage

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda', help='Device')
parser.add_argument('--gpu_id', type=int, default=0, help='Cuda GPU ID')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--seed', type=int, default=20,
                    help='Random seed for reproducibility')

start_time = time.time()
print("Start time: ")
print_local_time()

args = parser.parse_args()


def experiment(args):
    # Dataset
    # if os.path.isfile("../data_binaries/marco_query_generated/msmarco_flan_t5_xl_queries_generated/")

    dataset = QueryPassage(args)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    experiment(args)
