import time
import os.path
import torch
import os
import argparse
from experiment import Experiment
import wandb
from utils import *
from ablations import (
    ABLATION_PRESETS,
    DECOMPOSITION_TYPES,
    apply_ablation_preset,
    format_ablation_presets,
)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['WANDB_MODE'] = 'online'

parser = argparse.ArgumentParser()

parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--model_name', type=str, default='bert-base-uncased')

parser.add_argument('--expID', type=int, default=0)
parser.add_argument('--exp_name', type=str, default='my old story')
parser.add_argument('--max_steps', type=int, default=70000)
parser.add_argument('--lr', type=float, default=9e-5)
parser.add_argument('--lr_proj', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--output_dim', type=int, default=384)
parser.add_argument('--kl_weight', type=float, default=0.45)
parser.add_argument('--bce_weight', type=float, default=0.45)
parser.add_argument('--reg_weight', type=float, default=0.1)
parser.add_argument('--vol_margin', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--kl_margin', type=float, default=0.5)
parser.add_argument('--accumulation_steps', type=int, default=5)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--wandb', type=int, default=1)
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--rank', type=int, default=16)
parser.add_argument('--gaussian_eps', type=float, default=1e-6)
parser.add_argument('--decomposition_type', type=str,
                    default='stiefel_projected_lambdas',
                    choices=DECOMPOSITION_TYPES)
parser.add_argument('--ablation_preset', type=str, default='none',
                    choices=('none', *ABLATION_PRESETS.keys()))
parser.add_argument('--ablation_name', type=str, default='')
parser.add_argument('--stiefel_iterations', type=int, default=4)
parser.add_argument('--min_variance', type=float, default=1e-5)
parser.add_argument('--max_variance', type=float, default=10.0)
parser.add_argument('--list_ablation_presets', action='store_true')
parser.add_argument("--visualize_every", type=int, default=0,
                    help="Write query/passage geometry every N steps; 0 disables it.")
parser.add_argument("--visualization_dir", default="visualizations/finetune")
parser.add_argument("--visualization_examples", type=int, default=4)
parser.add_argument("--visualization_dpi", type=int, default=150)

args = parser.parse_args()
args = apply_ablation_preset(args)


def experiment(args):
    if args.list_ablation_presets:
        print(format_ablation_presets())
        return

    torch.set_float32_matmul_precision('high')
    args.cuda = True

    if args.wandb == 1:
        ablation_label = args.ablation_name or args.decomposition_type
        wandb.init(
            project='Gaussian IR Indexing',
            name=f'{args.exp_name}-{ablation_label}-{args.expID}-{args.model_name}-{args.seed}-{args.rank}-{args.output_dim}',
            config=args
        )

    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    print(args)
    set_seed(args.seed)

    exp = Experiment(args)
    exp.train()

    wandb.finish()


if __name__ == '__main__':

    experiment(args)
