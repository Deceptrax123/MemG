from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from agent.math_utils import GaussianMath, GeometricRetriever


class Node:
    def __init__(self, mean, scale=1.0, timestamp=0):
        self.mu = torch.tensor(mean, dtype=torch.float32)
        self.V = torch.eye(len(mean), 2) * scale
        self.d = torch.ones(len(mean)) * scale
        self.timestamp = timestamp

    def get_params(self):
        return self.mu, self.V, torch.ones(2), self.d


def test_geometric_energy_components_sum_to_total():
    query = Node([0, 0, 0, 0], scale=1)
    candidate = Node([1, 0, 0, 0], scale=2)
    mean, covariance, total = GaussianMath.geometric_energy_components(
        query.mu, query.V, query.d, candidate.mu, candidate.V, candidate.d
    )
    assert mean.item() == pytest.approx(1.0)
    assert covariance.item() > 0
    assert total.item() == pytest.approx(mean.item() + covariance.item())
    assert GaussianMath.geometric_energy(
        query.mu, query.V, query.d, candidate.mu, candidate.V, candidate.d
    ).item() == pytest.approx(total.item())


def test_reranker_scores_components_and_preserves_chronological_output():
    query = Node([0, 0, 0, 0], timestamp=0)
    close_late = Node([0.1, 0, 0, 0], timestamp=20)
    far_early = Node([2.0, 0, 0, 0], timestamp=10)
    scored = GeometricRetriever.score_candidates(query, [far_early, close_late])
    assert scored[0][1] is close_late
    assert scored[0][0] == pytest.approx(scored[0][2] + scored[0][3])
    output = GeometricRetriever.get_top_k_geometric(
        query, [far_early, close_late], k=2
    )
    assert output == [far_early, close_late]
