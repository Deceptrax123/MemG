import os
import time
import torch
from transformers import BertTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from data import load_data
from model import GaussianEmbeddingTuner
from torch.optim import AdamW
import gc
import wandb


os.environ['WANDB_MODE'] = 'online'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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
            visualize_every = getattr(self.args, "visualize_every", 0)
            if visualize_every > 0 and (step + 1) % visualize_every == 0:
                try:
                    from visualize_gaussians import visualize_projection_batch
                    self.model.eval()
                    with torch.no_grad():
                        q_mu, q_lambda, q_frame, q_diag, _ = self.model.gaussian_projection(encode_query)
                        p_mu, p_lambda, p_frame, p_diag, _ = self.model.gaussian_projection(encode_positives)
                    sample_count = min(getattr(self.args, "visualization_examples", 4), len(q_mu), len(p_mu))
                    means = torch.cat([q_mu[:sample_count], p_mu[:sample_count]])
                    eigenvalues = torch.cat([q_lambda[:sample_count], p_lambda[:sample_count]])
                    frames = torch.cat([q_frame[:sample_count], p_frame[:sample_count]])
                    diagonals = torch.cat([q_diag[:sample_count], p_diag[:sample_count]])
                    labels = [f"query {i + 1}" for i in range(sample_count)] + [f"passage {i + 1}" for i in range(sample_count)]
                    output = os.path.join(
                        getattr(self.args, "visualization_dir", "visualizations/finetune"),
                        f"step_{step + 1:07d}.png",
                    )
                    visualize_projection_batch(
                        output, labels=labels, means=means, lambdas=eigenvalues,
                        frames=frames, diagonal_matrices=diagonals,
                        title=f"Training geometry at step {step + 1}",
                        dpi=getattr(self.args, "visualization_dpi", 150),
                    )
                    print(f"Wrote training geometry to {output}")
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
                final_logs = logs | geometric_logs

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
