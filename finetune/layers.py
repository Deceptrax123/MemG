import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLP_Z(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(MLP_Z, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, output_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


class MLP_Gaussian_mu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Gaussian_mu, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)

        return x


class MLP_Gaussian_Stiefel_Sigma(nn.Module):
    def __init__(
        self,
        hidden_size,
        rank,
        eps=1e-6,
        decomposition_type='stiefel_projected_lambdas',
        stiefel_iterations=4,
        min_variance=1e-5,
        max_variance=10.0,
    ):
        super(MLP_Gaussian_Stiefel_Sigma, self).__init__()

        self.rank = rank
        self.eps = eps
        self.decomposition_type = decomposition_type
        self.stiefel_iterations = stiefel_iterations
        self.min_variance = min_variance
        self.max_variance = max_variance

        self.d_proj = nn.Linear(hidden_size, hidden_size)
        self.lambda_proj = nn.Linear(hidden_size, rank)

        self.A_proj = nn.Linear(hidden_size, hidden_size*rank)

    def _positive(self, values):
        values = torch.exp(values)
        if self.decomposition_type in {
            'vv_bounded_volumes',
            'diagonal_only',
        }:
            values = torch.clamp(values, self.min_variance, self.max_variance)
        return values

    def _project_to_stiefel(self, V_raw):
        V_ortho = V_raw
        I = torch.eye(self.rank, device=V_raw.device).unsqueeze(0)

        for _ in range(self.stiefel_iterations):
            V_t_V = torch.bmm(V_ortho.transpose(1, 2), V_ortho)
            V_ortho = torch.bmm(V_ortho, (1.5*I-0.5*V_t_V))

        return V_ortho

    def _vv_to_directions(self, V_raw):
        norms = torch.norm(V_raw, p=2, dim=1).clamp_min(self.eps)
        V = V_raw / norms.unsqueeze(1)
        lambdas = norms.pow(2)
        lambdas = torch.clamp(lambdas, self.min_variance, self.max_variance)

        return V, lambdas

    def forward(self, z):
        d = self._positive(self.d_proj(z))
        learned_lambdas = self._positive(self.lambda_proj(z))

        V_raw = torch.tanh(self.A_proj(z).view(
            z.size(0), z.size(1), self.rank))

        if self.decomposition_type == 'stiefel_projected_lambdas':
            V = self._project_to_stiefel(V_raw)
            lambdas = learned_lambdas
        elif self.decomposition_type == 'stiefel_projected_directions_only':
            V = self._project_to_stiefel(V_raw)
            lambdas = torch.ones(
                z.size(0), self.rank, device=z.device, dtype=z.dtype)
        elif self.decomposition_type == 'vv_bounded_volumes':
            V, lambdas = self._vv_to_directions(V_raw)
        elif self.decomposition_type == 'unprojected_lambdas':
            V = V_raw
            lambdas = learned_lambdas
        elif self.decomposition_type == 'diagonal_only':
            V = torch.zeros_like(V_raw)
            lambdas = torch.full(
                (z.size(0), self.rank), self.eps, device=z.device, dtype=z.dtype)
        else:
            raise ValueError(
                f"Unknown decomposition_type: {self.decomposition_type}")

        v_lambda = V*lambdas.unsqueeze(1)

        low_rank_cov = torch.bmm(v_lambda, V.transpose(1, 2))
        x_d = torch.diag_embed(d)

        x_sigma = x_d+low_rank_cov

        return V, lambdas, x_d, x_sigma


class MLP_Gaussian_sigma(nn.Module):
    def __init__(self, output_dim, rank, eps=1e-6):
        super(MLP_Gaussian_sigma, self).__init__()
        self.output_dim = output_dim
        self.rank = rank
        self.eps = eps
        self.factor_head = nn.Linear(output_dim, output_dim*self.rank)
        self.diag_head = nn.Linear(output_dim, output_dim)
        self.covariance_clipper = CovarianceClipper()

    def forward(self, x):
        x_v = self.factor_head(x)
        x_v = x_v.view(-1, self.output_dim, self.rank)

        d = torch.exp(self.diag_head(x))
        x_v, x_d = self.covariance_clipper(x_v, d)

        x_d = torch.diag_embed(x_d)
        x_sigma = torch.bmm(x_v, x_v.transpose(1, 2))+x_d

        return x_v, x_d, x_sigma


class CovarianceClipper(nn.Module):
    def __init__(self, min_var=1e-5, max_var=10):
        super(CovarianceClipper, self).__init__()

        self.min_var = min_var
        self.max_var = max_var

    def forward(self, v, d):
        d_clamped = torch.clamp(d, min=self.min_var, max=self.max_var)

        v_norms = torch.norm(v, p=2, dim=-1, keepdim=True)

        min_v_norm = torch.sqrt(torch.tensor(self.min_var, device=v.device))
        max_v_norm = torch.sqrt(torch.tensor(self.max_var, device=v.device))

        target_norms = torch.clamp(v_norms, min=min_v_norm, max=max_v_norm)

        scale_factor = target_norms / (v_norms + 1e-9)
        v_clamped = v * scale_factor

        return v_clamped, d_clamped
