import os
import torch
import argparse
from agent.helpers import load_model, load_tokenizer
from agent.locomo10_tester import LocomoTester
from agent.memory import GeoLogicMemorySystem

os.environ['WANDB_MODE'] = 'online'

parser = argparse.ArgumentParser()

parser.add_argument('--window_size', type=int, default=20)
parser.add_argument('--overlap_size', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--path', type=str, default='')
parser.add_argument('--expID', type=int, default=0)
parser.add_argument('--exp_name', type=str, default='my old story')
parser.add_argument('--learning_rate', type=float, default=0.05)
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
parser.add_argument('--alpha_success', type=float, default=1)
parser.add_argument('--beta_consistency', type=float, default=0.5)
parser.add_argument('--gamma_inertia', type=float, default=0.1)
parser.add_argument('--bc_weight', type=float, default=0.33)
parser.add_argument('--entity_weight', type=float, default=0.33)
parser.add_argument('--topic_weight', type=float, default=0.33)
parser.add_argument('--base_threshold', type=float, default=0.3)
parser.add_argument('--resume_idx', type=int, default=0)
parser.add_argument("--dataset_path", default="data/locomo10.json")
parser.add_argument("--interval", type=int, default=5)
parser.add_argument('--top_k', type=int, default=15)
parser.add_argument("--visualize_planning", "--visualize_query_answers",
                    dest="visualize_planning", action="store_true",
                    help="Write query, reference-answer, prediction, and low-rank frame diagnostics.")
parser.add_argument("--visualization_dir", default="visualizations/locomo_planning")
parser.add_argument("--visualization_max_plans", type=int, default=50)
parser.add_argument("--visualization_max_candidates", type=int, default=30)
parser.add_argument("--visualization_dpi", type=int, default=150)

# OpenAI generation settings
parser.add_argument('--openai_api_key', type=str, default=None,
                    help='OpenAI API key. Defaults to OPENAI_API_KEY from the environment.')
parser.add_argument('--openai_base_url', type=str, default=None,
                    help='Optional OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL if set.')
parser.add_argument('--gen_model_name', type=str, default='gpt-4.1-mini',
                    choices=['gpt-4.1-mini', 'gpt-4.1', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', 'o4-mini'])

args = parser.parse_args()


if __name__ == '__main__':


    torch.set_float32_matmul_precision('high')
    args.cuda = args.device == "cuda"
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
        torch.cuda.set_device(args.gpu_id)

    tokenizer = load_tokenizer(args)
    model = load_model(args, weights_path=args.path)

    system = GeoLogicMemorySystem(args, model, tokenizer)
    tester = LocomoTester(args, system, args.dataset_path)
    tester.run_test(resume_idx=args.resume_idx, interval=args.interval)
