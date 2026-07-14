from types import SimpleNamespace

import torch

from finetune.layers import MLP_Gaussian_Stiefel_Sigma
from finetune.model import GaussianEmbeddingTuner


def test_covariance_head_uses_constant_epsilon_identity():
    eps = 1e-4
    head = MLP_Gaussian_Stiefel_Sigma(
        hidden_size=6,
        rank=2,
        eps=eps,
        decomposition_type="stiefel_projected_lambdas",
    )
    z = torch.randn(3, 6, requires_grad=True)

    frame, lambdas, diagonal, covariance = head(z)

    expected_diagonal = torch.eye(6).expand(3, -1, -1) * eps
    expected_covariance = (
        torch.bmm(frame * lambdas.unsqueeze(1), frame.transpose(1, 2))
        + expected_diagonal
    )
    assert not hasattr(head, "d_proj")
    assert torch.equal(diagonal, expected_diagonal)
    assert torch.allclose(covariance, expected_covariance)


def test_identical_constant_d_cancels_in_bures_distance():
    eps = 1e-6
    model_stub = SimpleNamespace(args=SimpleNamespace(gaussian_eps=eps))
    means = torch.zeros(2, 3)
    frame = torch.tensor([
        [[1.0], [0.0], [0.0]],
        [[1.0], [0.0], [0.0]],
    ])
    lambdas = torch.ones(2, 1)
    diagonal = torch.full((2, 3), eps)

    distance = GaussianEmbeddingTuner.exact_bures_wasserstein(
        model_stub,
        means,
        frame,
        lambdas,
        diagonal,
        means,
        frame,
        lambdas,
        diagonal,
    )

    assert torch.allclose(distance, torch.zeros_like(distance), atol=1e-7)


def test_constant_diagonal_epsilon_must_be_positive():
    try:
        MLP_Gaussian_Stiefel_Sigma(hidden_size=4, rank=1, eps=0.0)
    except ValueError as error:
        assert "eps must be positive" in str(error)
    else:
        raise AssertionError("Expected non-positive eps to be rejected")
