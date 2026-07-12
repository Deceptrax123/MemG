import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from agent.visualize_planning import (
    PlanningVisualizer,
    directional_statistics,
    normal_density,
)
from finetune.visualize_gaussians import (
    covariance_in_basis,
    orthogonality_error,
    visualize_projection_batch,
)


def synthetic_geometry(batch=3, dim=32, rank=8):
    generator = torch.Generator().manual_seed(7)
    means = torch.randn(batch, dim, generator=generator)
    raw = torch.randn(batch, dim, rank, generator=generator)
    frames = torch.linalg.qr(raw, mode="reduced").Q
    lambdas = torch.exp(torch.randn(batch, rank, generator=generator))
    diagonals = torch.diag_embed(torch.exp(torch.randn(batch, dim, generator=generator)))
    return means, lambdas, frames, diagonals


def test_projected_covariance_and_orthonormal_frame():
    _, lambdas, frames, diagonals = synthetic_geometry()
    basis = np.eye(32)[:, :2]
    covariance = covariance_in_basis(frames[0], lambdas[0], diagonals[0], basis)
    error, gram = orthogonality_error(frames[0])
    assert covariance.shape == (2, 2)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert error < 1e-5
    assert np.allclose(gram, np.eye(8), atol=1e-5)


def test_directional_distribution_uses_full_low_rank_covariance():
    means, lambdas, frames, diagonals = synthetic_geometry()
    direction = frames[0, :, 0]
    mean, variance = directional_statistics(
        means[0], direction, means[1], frames[1], lambdas[1], diagonals[1]
    )
    x = np.linspace(mean - 4 * np.sqrt(variance), mean + 4 * np.sqrt(variance), 4000)
    density = normal_density(x, mean, variance)
    assert variance > 0
    assert np.trapezoid(density, x) == pytest.approx(1.0, abs=1e-3)


def test_query_passage_visualization_writes_png_and_json(tmp_path):
    means, lambdas, frames, diagonals = synthetic_geometry()
    output = tmp_path / "query_passage.png"
    diagnostics = visualize_projection_batch(
        output,
        labels=["query", "positive", "negative"],
        means=means,
        lambdas=lambdas,
        frames=frames,
        diagonal_matrices=diagonals,
    )
    assert output.exists() and output.stat().st_size > 0
    assert output.with_suffix(".json").exists()
    assert len(diagnostics["items"]) == 3


def test_query_answer_and_rank8_frame_reports(tmp_path):
    means, lambdas, frames, diagonals = synthetic_geometry()
    output = PlanningVisualizer(tmp_path).render_query_answer(
        question="Where did Caroline move from?",
        reference_answer="Caroline moved from Boston.",
        predicted_answer="Boston",
        plan={"difficulty_level": "single_hop"},
        means=means,
        lambdas=lambdas,
        frames=frames,
        diagonals=diagonals,
    )
    metadata = json.loads(output.with_suffix(".json").read_text())
    frame_output = output.with_name(f"{output.stem}_frames.png")
    assert output.exists() and output.stat().st_size > 0
    assert output.name == "gaussians.png"
    assert frame_output.exists() and frame_output.stat().st_size > 0
    assert len(metadata["states"]) == 3
    assert len(metadata["directional_distributions"]) == 8
    assert all(len(axis["states"]) == 3 for axis in metadata["directional_distributions"])
    assert "one d×8 Stiefel frame" in metadata["frame_note"]


def test_multi_gaussian_report_includes_queries_entities_and_answers(tmp_path):
    labels = [
        "raw query",
        "dense query",
        "hop 1",
        "target entity 1: Caroline",
        "target entity 2: Boston",
        "reference answer",
        "final answer",
    ]
    texts = [
        "Where did Caroline move from?",
        "Caroline's previous place of residence",
        "Caroline lived somewhere before moving",
        "Caroline",
        "Boston",
        "Caroline moved from Boston.",
        "Boston",
    ]
    means, lambdas, frames, diagonals = synthetic_geometry(batch=len(labels))
    output = PlanningVisualizer(tmp_path).render_query_answer(
        question=texts[0],
        reference_answer=texts[-2],
        predicted_answer=texts[-1],
        labels=labels,
        texts=texts,
        plan={"difficulty_level": "multi_hop"},
        means=means,
        lambdas=lambdas,
        frames=frames,
        diagonals=diagonals,
    )
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["labels"] == labels
    assert metadata["texts"] == texts
    assert len(metadata["states"]) == len(labels)
    assert all(
        len(axis["states"]) == len(labels)
        for axis in metadata["directional_distributions"]
    )


def test_wasserstein_faiss_hit_report_writes_hits_and_components(tmp_path):
    records = []
    for query_index, query_label in enumerate(["dense query", "hop 1"]):
        for hit_rank in range(1, 7):
            mean_energy = 0.5 * hit_rank + query_index
            std_energy = 0.2 * hit_rank
            records.append({
                "query_index": query_index,
                "query_label": query_label,
                "query_text": f"query variant {query_index}",
                "hit_rank": hit_rank,
                "faiss_index": hit_rank - 1,
                "node_id": f"mem_{hit_rank - 1}",
                "text": f"retrieved memory {hit_rank - 1}",
                "timestamp": 1000 + hit_rank,
                "topic": "test",
                "mean_energy": mean_energy,
                "std_profile_energy": std_energy,
                "wasserstein_squared": mean_energy + std_energy,
                "component_sum": mean_energy + std_energy,
            })
    output = PlanningVisualizer(tmp_path).render_retrieval_hits(
        question="Which memories are closest?",
        query_labels=["dense query", "hop 1"],
        records=records,
        max_candidates=5,
    )
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert output.name == "retrieval_wasserstein.png"
    assert output.exists() and output.stat().st_size > 0
    assert metadata["captured_hit_count"] == 12
    assert metadata["displayed_hit_count"] == 10
    assert "IndexFlatL2" in metadata["faiss_note"]
    assert all(
        record["wasserstein_squared"] == pytest.approx(record["component_sum"])
        for record in metadata["records"]
    )


def test_reranking_visualization_writes_energy_and_rank_report(tmp_path):
    records = []
    for index in range(12):
        selected = index < 4
        records.append({
            "candidate_index": index,
            "node_id": f"mem_{index}",
            "text": f"Candidate memory {index}",
            "timestamp": 1000 + index,
            "topic": "test",
            "stage1_mean_energy": 1.0 + index,
            "stage1_covariance_energy": 12.0 - index / 2,
            "stage1_total_energy": 13.0 + index / 2,
            "stage1_rank": index + 1,
            "passed_stage1": True,
            "stage2_mean_energy": 1.0 + index,
            "stage2_variance_energy": 0.25 * (12 - index),
            "stage2_total_energy": 4.0 + index,
            "stage2_rank": index + 1,
            "selected": selected,
            "selected_relevance_rank": index + 1 if selected else None,
            "chronological_output_position": 4 - index if selected else None,
        })
    output = PlanningVisualizer(tmp_path).render_reranking(
        question="Which memory should be selected?", records=records, max_candidates=10
    )
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert output.exists() and output.stat().st_size > 0
    assert output.name == "reranking.png"
    assert metadata["candidate_count"] == 12
    assert metadata["displayed_candidate_count"] == 10
    assert "chronologically" in metadata["ordering_note"]
    assert len(metadata["records"]) == 12
