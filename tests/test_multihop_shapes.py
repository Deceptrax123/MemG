import pytest

torch = pytest.importorskip("torch")

from geologic_memory.db import effective_covariance_diagonal


def test_effective_covariance_diagonal_supports_batched_multihop_queries():
    batch, dim, rank = 3, 1024, 8
    vectors = torch.ones(batch, dim, rank)
    eigenvalues = torch.full((batch, rank), 2.0)
    diagonal = torch.diag_embed(torch.full((batch, dim), 0.5))

    result = effective_covariance_diagonal(vectors, eigenvalues, diagonal)

    assert result.shape == (batch, dim)
    assert torch.allclose(result, torch.full((batch, dim), 16.5))


def test_effective_covariance_diagonal_rejects_rank_mismatch():
    with pytest.raises(ValueError, match="rank dimensions"):
        effective_covariance_diagonal(
            torch.ones(3, 16, 8),
            torch.ones(3, 7),
            torch.eye(16).repeat(3, 1, 1),
        )
