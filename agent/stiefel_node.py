import torch.nn as nn
import torch

DEVICE = 'cuda'


def make_stiefel_parameter(V_raw):
    with torch.no_grad():
        U, _, _ = torch.svd(V_raw, some=True)
    param = nn.Parameter(U)
    param.is_stiefel = True
    return param


class StiefelGaussianNode(nn.Module):
    def __init__(self, node_id, text, topic, entities, timestamp, mu_raw, V_raw, d_raw, topic_embed):
        super().__init__()
        self.node_id = node_id
        self.text = text
        self.timestamp = timestamp
        self.topic = topic
        self.entities = entities
        self.topic_embedding = topic_embed

        self.register_buffer("mu_orig", mu_raw.clone().detach())
        d_raw = d_raw.squeeze()

        if d_raw.dim() == 2:
            d_raw = torch.diagonal(d_raw)

        self.mu = nn.Parameter(mu_raw.clone())
        self.log_d = nn.Parameter(torch.log(d_raw.clone() + 1e-6))

        self.V = make_stiefel_parameter(V_raw)
        _, S, _ = torch.svd(V_raw, some=True)
        self.log_lam = nn.Parameter(torch.log(S**2 + 1e-6))

    def get_params(self):
        return self.mu, self.V, torch.exp(self.log_lam), torch.exp(self.log_d)

    def get_distribution(self, current_time=None, time_decay_rate=0.01):
        d = torch.exp(self.log_d)
        lam = torch.exp(self.log_lam)

        if current_time is not None:
            time_delta = max(0.0, current_time - self.timestamp)
            expansion = 1.0 + (time_decay_rate * time_delta)
            d = d * expansion
            lam = lam * expansion

        full_cov = (self.V @ torch.diag(lam) @ self.V.T) + torch.diag(d)

        full_cov = full_cov + \
            torch.eye(full_cov.shape[0], device=DEVICE) * 1e-5

        return torch.distributions.MultivariateNormal(self.mu, covariance_matrix=full_cov)
