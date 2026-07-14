import os
import time
import torch
from transformers import BertTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from data import load_data
from model import GaussianEmbeddingTuner
from torch.optim import AdamW
import gc
import wandb
import numpy as np


os.environ['WANDB_MODE'] = 'online'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def _wandb_histogram(values, max_values=100_000):
    """Build a bounded-size W&B histogram from a tensor or NumPy array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    values = np.asarray(values).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) > max_values:
        stride = int(np.ceil(len(values) / max_values))
        values = values[::stride][:max_values]
    return wandb.Histogram(values)


def _wandb_gaussian_distribution_logs(
    role, means, lambdas, frames, diagonal_matrices
):
    from visualize_gaussians import gaussian_distribution_values

    values = gaussian_distribution_values(
        means, lambdas, frames, diagonal_matrices
    )
    prefix = f"Gaussian distributions/{role}"
    return {
        f"{prefix}/mu components": _wandb_histogram(values["mu_components"]),
        f"{prefix}/mu norms": _wandb_histogram(values["mu_norms"]),
        f"{prefix}/sigma marginal variances": _wandb_histogram(
            values["sigma_marginal_variances"]
        ),
        f"{prefix}/sigma marginal standard deviations": _wandb_histogram(
            values["sigma_marginal_standard_deviations"]
        ),
        f"{prefix}/low-rank lambdas": _wandb_histogram(
            values["low_rank_lambdas"]
        ),
    }


class Experiment(object):
    def __init__(self, args):
        super(Experiment, self).__init__()
        self.args = args
        self.max_len = 8192
        self.args.device = 'cuda' if self.args.cuda else 'cpu'
        self.tokenizer = self._load_tokenizer_()
        self.accumulation_steps = self.args.accumulation_steps
        self.model = GaussianEmbeddingTuner(self.args)
        self.optimizer = self._select_optimizer_()

        self._set_device_()

        ablation_label = getattr(self.args, 'ablation_name', '') or getattr(
            self.args, 'decomposition_type', 'stiefel_projected_lambdas')
        self.exp_setting = "_".join(str(elem) for elem in [
                                    self.args.expID, ablation_label, self.args.seed, self.args.output_dim, self.args.model_name, self.args.rank, self.args.kl_weight, self.args.bce_weight, self.args.reg_weight])
        self.data_loader = load_data(self.args)
        self.scheduler = self._set_scheduler_()

    def _load_tokenizer_(self):
        print("Loading Tokenizer....")
        if self.args.model_name == 'bert-base-uncased':
            tokenizer = BertTokenizer.from_pretrained(
                '/home/models/bert-base-uncased')
            self.max_len = 512
        elif self.args.model_name == 'gte-large-en-v1.5':
            tokenizer = AutoTokenizer.from_pretrained(
                'Alibaba-NLP/gte-large-en-v1.5')
        elif self.args.model_name == 'nomic-embed-text-v1.5':
            tokenizer = AutoTokenizer.from_pretrained(
                '/home/models/bert-base-uncased')
            self.max_len = 8192
        elif self.args.model_name == 'all-MiniLM-L6-v2':
            tokenizer = AutoTokenizer.from_pretrained(
                f'sentence-transformers/{self.args.model_name}')
            self.max_len = 256
        elif self.args.model_name == 'Qwen3-Embedding-0.6B':
            tokenizer = AutoTokenizer.from_pretrained(
                '/home/models/qwen3-embedding-0.6b', padding_side='left')
            self.max_len = 8192
        elif self.args.model_name == 'bge-base-en-v1.5':
            tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
            self.max_len = 512

        return tokenizer

    def _select_optimizer_(self):
        parameters = [
            {"params": [p for n, p in self.model.named_parameters()], 'weight_decay': 0.0}]

        optimizer = AdamW(params=parameters,
                          lr=self.args.lr, eps=self.args.eps)
        return optimizer

    def _set_device_(self):
        if self.args.cuda:
            self.model = self.model.cuda()

    def _set_scheduler_(self):
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 500, self.args.max_steps)

        return scheduler

    def train(self, checkpoint=None, save_path=None):

        if checkpoint:
            self.model.load_state_dict(torch.load(f"{checkpoint}"))

        if save_path is None:
            traindir = os.path.join("../result/train", self.args.exp_name)
            checkpoint_step_dir = os.path.join(
                "../result/checkpoints", self.args.exp_name, self.exp_setting)

            if not os.path.exists(traindir):
                os.makedirs(traindir, exist_ok=True)
            if not os.path.exists(checkpoint_step_dir):
                os.makedirs(checkpoint_step_dir, exist_ok=True)

            save_path = os.path.join(
                "../result", "train", f"exp_model_{self.exp_setting}.checkpoint")

        for step, (queries, positives, negatives_list) in enumerate(self.data_loader):
            self.model.train()
            encode_query = self.tokenizer(
                list(queries), padding=True, truncation=True, return_tensors='pt', max_length=self.max_len
            ).to(self.args.device)

            encode_positives = self.tokenizer(
                list(positives), padding=True, truncation=True, return_tensors='pt', max_length=self.max_len
            ).to(self.args.device)

            flat_negatives = []

            if isinstance(negatives_list[0], (tuple, list)):
                batch_size = len(queries)
                num_negs = len(negatives_list)

                for b in range(batch_size):
                    for n in range(num_negs):
                        flat_negatives.append(negatives_list[n][b])
            else:
                flat_negatives = list(negatives_list)

            encode_negatives = self.tokenizer(
                flat_negatives, padding=True, truncation=True, return_tensors='pt', max_length=self.max_len
            ).to(self.args.device)
            self.optimizer.zero_grad()

            loss, kl_loss, bce_loss, reg_loss, qv, ql, qd, pv, pl, pd = self.model(
                encode_query, encode_positives, encode_negatives)
            loss = loss/self.accumulation_steps

            loss.backward()
            if not step % self.accumulation_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.scheduler.step()
            visualization_logs = {}
            visualize_every = getattr(self.args, "visualize_every", 0)
            wandb_plot_every = getattr(self.args, "wandb_plot_every", 0)
            local_plot_due = (
                visualize_every > 0 and (step + 1) % visualize_every == 0
            )
            wandb_plot_due = (
                self.args.wandb == 1
                and wandb_plot_every > 0
                and (step + 1) % wandb_plot_every == 0
            )
            if local_plot_due or wandb_plot_due:
                try:
                    from visualize_gaussians import visualize_projection_batch

                    sample_count = min(
                        getattr(self.args, "visualization_examples", 4),
                        len(queries),
                        len(positives),
                    )
                    pair_indices = torch.randperm(len(queries))[
                        :sample_count
                    ].tolist()
                    sampled_queries = [queries[index] for index in pair_indices]
                    sampled_positives = [positives[index] for index in pair_indices]
                    if negatives_list and isinstance(
                        negatives_list[0], (tuple, list)
                    ):
                        negative_pool = [
                            negative
                            for query_negatives in negatives_list
                            for negative in query_negatives
                        ]
                    else:
                        negative_pool = list(negatives_list)
                    negative_count = min(sample_count, len(negative_pool))
                    negative_indices = torch.randperm(len(negative_pool))[
                        :negative_count
                    ].tolist()
                    sampled_negatives = [
                        negative_pool[index] for index in negative_indices
                    ]
                    diagnostic_texts = [
                        *sampled_queries,
                        *sampled_positives,
                        *sampled_negatives,
                    ]
                    diagnostic_inputs = self.tokenizer(
                        diagnostic_texts,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_len,
                    ).to(self.args.device)
                    self.model.eval()
                    with torch.no_grad():
                        means, eigenvalues, frames, diagonals, _ = (
                            self.model.gaussian_projection(diagnostic_inputs)
                        )
                    labels = (
                        [f"query {i + 1}" for i in range(sample_count)]
                        + [f"positive {i + 1}" for i in range(sample_count)]
                        + [f"negative {i + 1}" for i in range(negative_count)]
                    )
                    groups = (
                        ["query"] * sample_count
                        + ["positive"] * sample_count
                        + ["negative"] * negative_count
                    )
                    output = os.path.join(
                        getattr(self.args, "visualization_dir", "visualizations/finetune"),
                        f"step_{step + 1:07d}.png",
                    )
                    visualize_projection_batch(
                        output, labels=labels, means=means, lambdas=eigenvalues,
                        frames=frames, diagonal_matrices=diagonals,
                        groups=groups,
                        title=f"Training geometry at step {step + 1}",
                        dpi=getattr(self.args, "visualization_dpi", 150),
                    )
                    print(f"Wrote training geometry to {output}")
                    if wandb_plot_due:
                        visualization_logs[
                            "Gaussian geometry/query-positive-negative ellipses"
                        ] = wandb.Image(
                            output,
                            caption=(
                                f"Random dataset samples at step {step + 1}; "
                                "ellipses show projected 2σ learned covariance."
                            ),
                        )
                        role_offsets = {
                            "Query": (0, sample_count),
                            "Positive passage": (
                                sample_count, 2 * sample_count
                            ),
                            "Negative passage": (
                                2 * sample_count,
                                2 * sample_count + negative_count,
                            ),
                        }
                        for role, (start, end) in role_offsets.items():
                            if end > start:
                                visualization_logs.update(
                                    _wandb_gaussian_distribution_logs(
                                        role,
                                        means[start:end],
                                        eigenvalues[start:end],
                                        frames[start:end],
                                        diagonals[start:end],
                                    )
                                )
                    del diagnostic_inputs
                except Exception as exc:
                    print(f"Visualization failed at step {step + 1}: {exc}")
            del encode_query, encode_negatives, encode_positives
            torch.cuda.empty_cache()
            gc.collect()

            self.model.eval()
            with torch.no_grad():
                logs = {
                    'Step': step,
                    'Loss': loss,
                    'Contrastive Loss': kl_loss,
                    'Regularizer': reg_loss,
                }

                geometric_logs = self.evaluate(qv, ql, qd, pv, pl, pd)
                final_logs = logs | geometric_logs | visualization_logs

                if self.args.wandb == 1:
                    wandb.log(final_logs)
                print(final_logs)

                torch.save(self.model.state_dict(), os.path.join(
                    traindir, f"experiment_{self.exp_setting}.checkpoint"))
                if (step+1) % 500 == 0:
                    torch.save(self.model.state_dict(), os.path.join(
                        checkpoint_step_dir, f"experiment_{step+1}_{self.exp_setting}.checkpoint"))
                if step+1 == self.args.max_steps:
                    print("Training complete!!!")
                    break

    def evaluate(self, qv, ql, qd, pv, pl, pd):
        query_volume = self.model._get_logdet(qv, ql, qd).mean()
        passage_volume = self.model._get_logdet(pv, pl, pd).mean()

        volume_containment = passage_volume-query_volume

        logs = {
            'Query Volume': query_volume.item(),
            'Passage Volume': passage_volume.item(),
            'Consistency': volume_containment.item(),
            'Norm_V': torch.norm(qv).item(),
            'Norm D': torch.norm(qd).item(),

            'Query Lambda/Mean': ql.mean().item(),
            'Query Lambda/Max': ql.max().item(),
            'Query Lambda/Min': ql.min().item(),

            'Passage Lambda/Mean': pl.mean().item(),
            'Passage Lambda/Max': pl.max().item(),
            'Passage Lambda/Min': pl.min().item(),
        }
        logs['Query Lambda Distribution'] = wandb.Histogram(
            ql.detach().cpu().numpy())
        logs['Passage Lambda Distribution'] = wandb.Histogram(
            pl.detach().cpu().numpy())
        return logs
