import numpy as np

from finetune.offline_z_analysis import Geometry
from finetune.visualize_arxiv_topics import (
    _normalized_gaussian_product,
    _plot_paper_gaussian_manifold,
    _project_paper_covariances,
    synthetic_topic_geometry,
)


def test_normalized_gaussian_product_uses_precision_weighted_mean():
    means = np.asarray([[0.0, 2.0], [4.0, 2.0]])
    covariances = np.asarray([
        np.diag([1.0, 3.0]),
        np.diag([4.0, 3.0]),
    ])

    mean, covariance = _normalized_gaussian_product(means, covariances)

    assert np.allclose(mean, [0.8, 2.0])
    assert np.allclose(covariance, np.diag([1.6, 3.0]))


def test_project_paper_covariances_uses_learned_diagonal_and_frame():
    geometry = Geometry(
        labels=["query", "paper"],
        texts=["query", "abstract"],
        positive_indices=[1],
        z=np.zeros((2, 3)),
        means=np.zeros((2, 3)),
        lambdas=np.asarray([[5.0], [5.0]]),
        frames=np.asarray([
            [[1.0], [0.0], [0.0]],
            [[1.0], [0.0], [0.0]],
        ]),
        diagonals=np.asarray([
            [2.0, 3.0, 7.0],
            [2.0, 3.0, 7.0],
        ]),
    ).validate()
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    projected = _project_paper_covariances(geometry, basis)

    assert projected.shape == (2, 2, 2)
    assert np.allclose(projected[0], np.diag([7.0, 3.0]))


def test_paper_gaussian_manifold_has_no_topic_aggregation(tmp_path):
    records = [
        {"arxiv_id": "p1", "topic": "physics", "text": "physics abstract"},
        {"arxiv_id": "p2", "topic": "chemistry", "text": "chemistry abstract"},
        {"arxiv_id": "p3", "topic": "biology", "text": "biology abstract"},
    ]
    geometry = synthetic_topic_geometry(records, dim=12, rank=2, seed=7)
    output = tmp_path / "paper_gaussian_manifold.png"

    report = _plot_paper_gaussian_manifold(
        output, records, geometry, standard_deviations=2.0, dpi=45
    )

    assert output.exists() and output.stat().st_size > 0
    assert report["aggregation"] == "none"
    assert len(report["items"]) == len(records)
    assert all(item["axis_lengths"][0] > 0 for item in report["items"])
