"""Visualize whether learned Gaussian embeddings organize arXiv topics.

The global manifold uses PCA of learned means as its display plane. Topic
ellipses are normalized products of the paper Gaussians projected into that
plane; they are not empirical covariance contours of the displayed dots.
Every paper also gets an independent local top-covariance plane, and topic
sheets explicitly use unshared subplot axes. Orthonormal frames are analyzed
intrinsically through Gram matrices, eigenvalue profiles, and pairwise
principal-angle similarity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, BertTokenizer

from finetune.model import GaussianEmbeddingTuner
from finetune.offline_z_analysis import (
    Geometry,
    _array,
    _matplotlib,
    _plot_learned_ellipses,
    learned_local_ellipse_planes,
    load_geometry,
    sample_gaussians,
    save_geometry,
)


DEFAULT_DATASET = Path(__file__).with_name("arxiv_topic_abstracts.json")
TOPIC_COLORS = {
    "physics": "#4c78a8",
    "chemistry": "#f58518",
    "biology": "#54a24b",
}


def load_topic_records(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate the sourced topic JSON, returning flattened papers."""
    path = Path(path)
    payload = json.loads(path.read_text())
    records: list[dict[str, Any]] = []
    for topic_entry in payload.get("topics", []):
        topic = topic_entry.get("topic")
        if not topic or not topic_entry.get("papers"):
            raise ValueError("Each topic needs a name and at least one paper")
        for paper in topic_entry["papers"]:
            required = {"arxiv_id", "title", "abstract", "url"}
            missing = required.difference(paper)
            if missing:
                raise ValueError(
                    f"Paper in {topic} is missing: {', '.join(sorted(missing))}"
                )
            records.append({
                **paper,
                "topic": topic,
                "topic_display_name": topic_entry.get("display_name", topic.title()),
                "text": f"{paper['title']}. {paper['abstract']}",
            })
    if len({record["arxiv_id"] for record in records}) != len(records):
        raise ValueError("arXiv IDs must be unique")
    if len({record["topic"] for record in records}) < 2:
        raise ValueError("At least two topics are required for manifold analysis")
    return payload, records


def synthetic_topic_geometry(
    records: list[dict[str, Any]],
    dim: int = 48,
    rank: int = 6,
    seed: int = 19,
) -> Geometry:
    """Construct clustered geometry for offline testing of the report."""
    rng = np.random.default_rng(seed)
    topics = list(dict.fromkeys(record["topic"] for record in records))
    topic_centers = {}
    topic_frames = {}
    for topic_index, topic in enumerate(topics):
        center = rng.normal(size=dim)
        center /= np.linalg.norm(center)
        topic_centers[topic] = center * (4.0 + topic_index)
        topic_frames[topic] = np.linalg.qr(
            rng.normal(size=(dim, rank)), mode="reduced"
        )[0]

    means = []
    frames = []
    lambdas = []
    diagonals = []
    z = []
    for record in records:
        topic = record["topic"]
        mean = topic_centers[topic] + rng.normal(scale=0.18, size=dim)
        frame = np.linalg.qr(
            topic_frames[topic] + rng.normal(scale=0.035, size=(dim, rank)),
            mode="reduced",
        )[0]
        means.append(mean)
        z.append(mean + rng.normal(scale=0.2, size=dim))
        frames.append(frame)
        lambdas.append(rng.lognormal(mean=-0.6, sigma=0.18, size=rank))
        diagonals.append(rng.lognormal(mean=-2.0, sigma=0.15, size=dim))
    return Geometry(
        labels=[record["arxiv_id"] for record in records],
        texts=[record["text"] for record in records],
        positive_indices=[1],
        z=np.asarray(z),
        means=np.asarray(means),
        lambdas=np.asarray(lambdas),
        frames=np.asarray(frames),
        diagonals=np.asarray(diagonals),
    ).validate()


def _model_and_tokenizer(args: argparse.Namespace):
    model_args = SimpleNamespace(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        rank=args.rank,
        gaussian_eps=args.gaussian_eps,
        decomposition_type=args.decomposition_type,
        stiefel_iterations=args.stiefel_iterations,
        min_variance=args.min_variance,
        max_variance=args.max_variance,
    )
    if args.model_name == "bert-base-uncased":
        tokenizer_id = args.tokenizer_path or "/home/models/bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(tokenizer_id)
    else:
        defaults = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
            "Qwen3-Embedding-0.6B": "/home/models/qwen3-embedding-0.6b",
        }
        tokenizer_id = args.tokenizer_path or defaults.get(args.model_name, args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=args.trust_remote_code
        )
    model = GaussianEmbeddingTuner(model_args).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def checkpoint_topic_geometry(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> Geometry:
    """Encode all paper title/abstract texts with a finetuned checkpoint."""
    model, tokenizer = _model_and_tokenizer(args)
    max_length = {
        "bert-base-uncased": 512,
        "all-MiniLM-L6-v2": 256,
        "bge-base-en-v1.5": 512,
    }.get(args.model_name, 8192)
    collected: dict[str, list[np.ndarray]] = {
        "z": [], "means": [], "lambdas": [], "frames": [], "diagonals": [],
    }
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            encoded = tokenizer(
                [record["text"] for record in batch], padding=True,
                truncation=True, max_length=max_length, return_tensors="pt",
            ).to(args.device)
            encoder_embedding = model.get_cls(encoded)
            z = model.z_transform(encoder_embedding)
            means = model.center_prediction(z)
            frames, lambdas, diagonal_matrices, _ = model.decomposed_covariance(z)
            collected["z"].append(_array(z))
            collected["means"].append(_array(means))
            collected["lambdas"].append(_array(lambdas))
            collected["frames"].append(_array(frames))
            collected["diagonals"].append(_array(diagonal_matrices))
    return Geometry(
        labels=[record["arxiv_id"] for record in records],
        texts=[record["text"] for record in records],
        positive_indices=[1],
        z=np.concatenate(collected["z"]),
        means=np.concatenate(collected["means"]),
        lambdas=np.concatenate(collected["lambdas"]),
        frames=np.concatenate(collected["frames"]),
        diagonals=np.concatenate(collected["diagonals"]),
    ).validate()


def pairwise_mean_squared(means: np.ndarray) -> np.ndarray:
    differences = means[:, None, :] - means[None, :, :]
    return np.sum(differences**2, axis=-1)


def pairwise_model_bures(geometry: Geometry) -> np.ndarray:
    """Pairwise version of the finetune model's covariance-aware surrogate."""
    count = len(geometry.labels)
    distances = np.zeros((count, count), dtype=np.float64)
    traces = geometry.lambdas.sum(axis=1) + geometry.diagonals.sum(axis=1)
    for first in range(count):
        for second in range(first + 1, count):
            mean_energy = np.sum((geometry.means[first] - geometry.means[second]) ** 2)
            diagonal_cross = np.sqrt(
                np.clip(
                    geometry.diagonals[first] * geometry.diagonals[second],
                    0.0,
                    None,
                )
            ).sum()
            frame_cross = geometry.frames[first].T @ geometry.frames[second]
            scaled_cross = (
                np.sqrt(geometry.lambdas[first])[:, None]
                * frame_cross
                * np.sqrt(geometry.lambdas[second])[None, :]
            )
            low_rank_cross = np.linalg.svd(scaled_cross, compute_uv=False).sum()
            covariance_energy = traces[first] + traces[second] - 2.0 * (
                diagonal_cross + low_rank_cross
            )
            distances[first, second] = distances[second, first] = max(
                0.0, float(mean_energy + covariance_energy)
            )
    return distances


def pairwise_frame_similarity(frames: np.ndarray) -> np.ndarray:
    """Mean cosine of principal angles between learned frame subspaces."""
    count = len(frames)
    similarities = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            singular_values = np.linalg.svd(
                frames[first].T @ frames[second], compute_uv=False
            )
            similarity = float(np.clip(singular_values, 0.0, 1.0).mean())
            similarities[first, second] = similarities[second, first] = similarity
    return similarities


def cluster_metrics(matrix: np.ndarray, topics: list[str]) -> dict[str, Any]:
    """Compute full-space within/between separation and silhouette metrics."""
    topics_array = np.asarray(topics)
    within = []
    between = []
    silhouettes = []
    for first in range(len(topics)):
        same = (topics_array == topics_array[first])
        same[first] = False
        other_topics = [topic for topic in dict.fromkeys(topics) if topic != topics[first]]
        within.extend(matrix[first, same].tolist())
        between.extend(matrix[first, ~same & (np.arange(len(topics)) != first)].tolist())
        if not np.any(same):
            continue
        a_value = float(matrix[first, same].mean())
        b_value = min(
            float(matrix[first, topics_array == topic].mean()) for topic in other_topics
        )
        denominator = max(a_value, b_value, 1e-12)
        silhouettes.append((b_value - a_value) / denominator)
    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))
    return {
        "within_topic_mean": within_mean,
        "between_topic_mean": between_mean,
        "between_to_within_ratio": between_mean / max(within_mean, 1e-12),
        "mean_silhouette": float(np.mean(silhouettes)),
        "per_item_silhouette": silhouettes,
    }


def _topic_boundaries(topics: list[str]) -> list[int]:
    return [
        index for index in range(1, len(topics)) if topics[index] != topics[index - 1]
    ]


def _project_paper_covariances(
    geometry: Geometry, basis: np.ndarray
) -> np.ndarray:
    """Project each learned low-rank-plus-diagonal covariance into a 2D basis."""
    projected = []
    for frame, lambdas, diagonal in zip(
        geometry.frames, geometry.lambdas, geometry.diagonals
    ):
        diagonal_part = (basis.T * diagonal[None, :]) @ basis
        projected_frame = basis.T @ frame
        low_rank_part = (
            projected_frame * lambdas[None, :]
        ) @ projected_frame.T
        covariance = diagonal_part + low_rank_part
        projected.append((covariance + covariance.T) / 2.0)
    return np.asarray(projected)


def _normalized_gaussian_product(
    means: np.ndarray, covariances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Precision-average Gaussians without sample-count-driven PoE shrinkage."""
    precisions = np.linalg.inv(covariances)
    aggregate_precision = precisions.mean(axis=0)
    aggregate_covariance = np.linalg.inv(aggregate_precision)
    information_means = np.einsum("nij,nj->ni", precisions, means)
    aggregate_mean = aggregate_covariance @ information_means.mean(axis=0)
    return aggregate_mean, (aggregate_covariance + aggregate_covariance.T) / 2.0


def _plot_topic_manifold(
    output: Path,
    records: list[dict[str, Any]],
    geometry: Geometry,
    topic_ellipse_standard_deviations: float,
    dpi: int,
) -> dict[str, Any]:
    """Plot learned means and precision-aggregated topic Gaussians."""
    plt = _matplotlib()
    means = geometry.means
    centered = means - means.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    projected = centered @ basis
    projected_covariances = _project_paper_covariances(geometry, basis)
    total_energy = np.sum(singular_values**2)
    explained = (singular_values[:2] ** 2 / max(total_energy, 1e-12)).tolist()
    figure, (mean_axis, gaussian_axis) = plt.subplots(
        1, 2, figsize=(17, 7), constrained_layout=True
    )
    topics = list(dict.fromkeys(record["topic"] for record in records))
    topic_ellipses = []
    circle = np.stack([
        np.cos(np.linspace(0.0, 2.0 * np.pi, 361)),
        np.sin(np.linspace(0.0, 2.0 * np.pi, 361)),
    ])
    for topic in topics:
        indices = [i for i, record in enumerate(records) if record["topic"] == topic]
        points = projected[indices]
        color = TOPIC_COLORS.get(topic, "#777777")
        centroid, covariance = _normalized_gaussian_product(
            points, projected_covariances[indices]
        )
        values, vectors = np.linalg.eigh(covariance)
        values = np.clip(values, 1e-12, None)
        axis_lengths = topic_ellipse_standard_deviations * np.sqrt(values)
        contour = centroid[:, None] + vectors @ np.diag(axis_lengths) @ circle
        gaussian_axis.fill(
            contour[0], contour[1], color=color, alpha=0.10, label=topic.title()
        )
        gaussian_axis.plot(
            contour[0], contour[1], color=color, linewidth=2.0, alpha=0.9
        )
        topic_ellipses.append({
            "topic": topic,
            "paper_count": len(indices),
            "aggregate_mean": centroid.tolist(),
            "aggregate_covariance": covariance.tolist(),
            "axis_lengths": axis_lengths.tolist(),
            "standard_deviations": topic_ellipse_standard_deviations,
            "aggregation": "normalized_product_of_projected_learned_gaussians",
        })
        mean_axis.scatter(
            points[:, 0], points[:, 1], s=70, color=color, label=topic.title()
        )
        mean_axis.scatter(
            *centroid, marker="X", s=180, color=color, edgecolor="black"
        )
        gaussian_axis.scatter(
            *centroid, marker="X", s=180, color=color, edgecolor="black"
        )
        for index in indices:
            mean_axis.plot(
                [projected[index, 0], centroid[0]],
                [projected[index, 1], centroid[1]],
                color=color, alpha=0.22, linewidth=1,
            )
            mean_axis.annotate(
                records[index]["arxiv_id"], projected[index],
                xytext=(4, 4), textcoords="offset points", fontsize=8,
            )
    mean_axis.set(
        title="Learned mean metric (PCA zoom)",
        xlabel=f"mean PCA 1 ({explained[0] * 100:.1f}% variance)",
        ylabel=f"mean PCA 2 ({explained[1] * 100:.1f}% variance)",
    )
    gaussian_axis.set(
        title="Normalized learned-Gaussian aggregates (true scale)",
        xlabel="mean PCA 1",
        ylabel="mean PCA 2",
        aspect="equal",
    )
    figure.suptitle(
        "Learned Gaussian topic geometry — precision-weighted aggregation",
        fontsize=15,
    )
    figure.text(
        0.5, 0.01,
        "Large outlines are normalized products of projected learned Gaussians; "
        "they are not empirical spread contours.",
        ha="center", fontsize=9, alpha=0.75,
    )
    for plot_axis in (mean_axis, gaussian_axis):
        plot_axis.grid(alpha=0.2)
        plot_axis.legend()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return {
        "explained_variance_ratio": explained,
        "projected_means": projected.tolist(),
        "topic_ellipses": topic_ellipses,
        "topic_ellipse_note": (
            "Each topic is a normalized product of its projected learned paper "
            "Gaussians: precisions are averaged and means are precision weighted."
        ),
        "display_note": (
            "The left panel zooms to learned means; the right panel preserves the "
            "true scale of aggregate learned covariance."
        ),
    }


def _plot_paper_gaussian_manifold(
    output: Path,
    records: list[dict[str, Any]],
    geometry: Geometry,
    standard_deviations: float,
    dpi: int,
) -> dict[str, Any]:
    """Overlay every paper's own learned mean and covariance in one PCA plane."""
    plt = _matplotlib()
    centered = geometry.means - geometry.means.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    projected_means = centered @ basis
    projected_covariances = _project_paper_covariances(geometry, basis)
    total_energy = np.sum(singular_values**2)
    explained = (singular_values[:2] ** 2 / max(total_energy, 1e-12)).tolist()
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    seen_topics: set[str] = set()
    items = []

    for index, record in enumerate(records):
        covariance = projected_covariances[index]
        values, vectors = np.linalg.eigh(covariance)
        values = np.clip(values, 1e-12, None)
        axis_lengths = standard_deviations * np.sqrt(values)
        contour = (
            projected_means[index, :, None]
            + vectors @ np.diag(axis_lengths) @ circle
        )
        topic = record["topic"]
        color = TOPIC_COLORS.get(topic, "#777777")
        label = topic.title() if topic not in seen_topics else None
        seen_topics.add(topic)
        axis.fill(contour[0], contour[1], color=color, alpha=0.018)
        axis.plot(
            contour[0], contour[1], color=color, alpha=0.42,
            linewidth=0.9, label=label, zorder=1,
        )
        axis.scatter(
            *projected_means[index], color=color, edgecolor="black",
            linewidth=0.35, s=38, zorder=3,
        )
        axis.annotate(
            record["arxiv_id"], projected_means[index],
            xytext=(4, 4), textcoords="offset points", fontsize=7, zorder=4,
        )
        items.append({
            "arxiv_id": record["arxiv_id"],
            "topic": topic,
            "projected_mean": projected_means[index].tolist(),
            "projected_learned_covariance": covariance.tolist(),
            "axis_lengths": axis_lengths.tolist(),
        })

    axis.set(
        title="Non-aggregated learned paper Gaussians in the mean-PCA plane",
        xlabel=f"mean PCA 1 ({explained[0] * 100:.1f}% variance)",
        ylabel=f"mean PCA 2 ({explained[1] * 100:.1f}% variance)",
        aspect="equal",
    )
    axis.text(
        0.01, 0.01,
        "Every point is one learned μ; every outline is that same paper's learned "
        "covariance. No topic aggregation.",
        transform=axis.transAxes, fontsize=9, alpha=0.8,
    )
    axis.grid(alpha=0.2)
    axis.legend(title="Abstract topic")
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return {
        "explained_variance_ratio": explained,
        "standard_deviations": standard_deviations,
        "coordinate_system": "shared PCA plane fitted to learned paper means",
        "aggregation": "none",
        "items": items,
    }


def _plot_distance_matrices(
    output: Path,
    records: list[dict[str, Any]],
    mean_distances: np.ndarray,
    bures_distances: np.ndarray,
    dpi: int,
) -> None:
    plt = _matplotlib()
    labels = [record["arxiv_id"] for record in records]
    topics = [record["topic"] for record in records]
    figure, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0], mean_distances, "Full-space learned-mean squared distance"),
        (axes[1], bures_distances, "Model covariance-aware squared distance"),
    ):
        image = axis.imshow(matrix, cmap="magma")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        axis.set_title(title)
        for boundary in _topic_boundaries(topics):
            axis.axhline(boundary - 0.5, color="white", linewidth=1.5)
            axis.axvline(boundary - 0.5, color="white", linewidth=1.5)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def _plot_frame_diagnostics(
    output_dir: Path,
    records: list[dict[str, Any]],
    geometry: Geometry,
    dpi: int,
) -> list[dict[str, Any]]:
    plt = _matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for index, record in enumerate(records):
        frame = geometry.frames[index]
        gram = frame.T @ frame
        error = float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro"))
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        image = axes[0].imshow(gram, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[0].set(
            title=f"Frame Gram matrix; error={error:.2e}",
            xlabel="frame direction", ylabel="frame direction",
        )
        figure.colorbar(image, ax=axes[0], fraction=0.046)
        axes[1].bar(
            np.arange(1, len(geometry.lambdas[index]) + 1),
            geometry.lambdas[index], color=TOPIC_COLORS.get(record["topic"], "#777777"),
        )
        axes[1].set(
            title="Variance attached to each frame direction",
            xlabel="local frame direction", ylabel="lambda",
        )
        axes[1].grid(axis="y", alpha=0.2)
        figure.suptitle(
            f"{record['arxiv_id']} — {record['topic'].title()} local orthonormal frame"
        )
        filename = f"{index:03d}_{record['arxiv_id'].replace('.', '_')}.png"
        figure.savefig(output_dir / filename, dpi=dpi)
        plt.close(figure)
        diagnostics.append({
            "arxiv_id": record["arxiv_id"],
            "topic": record["topic"],
            "file": f"frames/{filename}",
            "orthogonality_error_fro": error,
            "gram_diagonal_mean": float(np.diag(gram).mean()),
            "lambda_values": geometry.lambdas[index].tolist(),
        })
    return diagnostics


def _plot_frame_similarity(
    output: Path,
    records: list[dict[str, Any]],
    similarities: np.ndarray,
    dpi: int,
) -> None:
    plt = _matplotlib()
    labels = [record["arxiv_id"] for record in records]
    topics = [record["topic"] for record in records]
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(similarities, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    axis.set_title("Frame-subspace similarity: mean cosine of principal angles")
    for boundary in _topic_boundaries(topics):
        axis.axhline(boundary - 0.5, color="white", linewidth=1.5)
        axis.axvline(boundary - 0.5, color="white", linewidth=1.5)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def _plot_topic_ellipse_sheets(
    output_dir: Path,
    records: list[dict[str, Any]],
    geometry: Geometry,
    clouds: np.ndarray,
    planes: np.ndarray,
    variances: np.ndarray,
    standard_deviations: float,
    dpi: int,
) -> list[str]:
    """Draw topic sheets whose subplots each retain independent local axes."""
    plt = _matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    files = []
    for topic in dict.fromkeys(record["topic"] for record in records):
        indices = [i for i, record in enumerate(records) if record["topic"] == topic]
        columns = 2
        rows = math.ceil(len(indices) / columns)
        figure, axes = plt.subplots(
            rows, columns, figsize=(6.2 * columns, 5.3 * rows),
            constrained_layout=True, squeeze=False,
        )
        for subplot, index in zip(axes.flat, indices):
            samples_2d = (clouds[index] - geometry.means[index]) @ planes[index]
            subplot.scatter(
                samples_2d[:, 0], samples_2d[:, 1], s=8, alpha=0.16,
                color=TOPIC_COLORS.get(topic, "#777777"),
            )
            for level, style in ((1.0, "--"), (standard_deviations, "-")):
                contour = np.sqrt(variances[index])[:, None] * circle * level
                subplot.plot(contour[0], contour[1], linestyle=style, linewidth=2)
            subplot.scatter(0.0, 0.0, marker="x", color="black", s=55)
            limit = max(
                float(standard_deviations * np.sqrt(variances[index]).max()),
                float(np.quantile(np.abs(samples_2d), 0.995)),
            ) * 1.15
            subplot.set_xlim(-limit, limit)
            subplot.set_ylim(-limit, limit)
            subplot.set_aspect("equal", adjustable="box")
            subplot.set(
                title=records[index]["arxiv_id"],
                xlabel="own covariance direction 1 from μ",
                ylabel="own covariance direction 2 from μ",
            )
            subplot.grid(alpha=0.2)
        for subplot in axes.flat[len(indices):]:
            subplot.set_visible(False)
        figure.suptitle(
            f"{topic.title()} — separate learned-covariance ellipse panels "
            "(subplot axes are not shared)"
        )
        filename = f"{topic}.png"
        figure.savefig(output_dir / filename, dpi=dpi)
        plt.close(figure)
        files.append(f"topic_ellipses/{filename}")
    return files


def analyze_topic_geometry(
    dataset: dict[str, Any],
    records: list[dict[str, Any]],
    geometry: Geometry,
    output_dir: str | Path,
    *,
    samples: int = 256,
    ellipse_standard_deviations: float = 2.0,
    topic_ellipse_standard_deviations: float = 2.0,
    seed: int = 20,
    dpi: int = 150,
) -> dict[str, Any]:
    """Generate topic manifold, local ellipse, and frame diagnostics."""
    geometry = geometry.validate()
    if len(records) != len(geometry.labels):
        raise ValueError("Topic records and geometry must have the same item count")
    expected_labels = [record["arxiv_id"] for record in records]
    if geometry.labels != expected_labels:
        raise ValueError("Geometry labels must match arXiv IDs in dataset order")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    topics = [record["topic"] for record in records]

    clouds = sample_gaussians(geometry, samples, rng)
    planes, variances, residuals = learned_local_ellipse_planes(geometry)
    ellipse_items = _plot_learned_ellipses(
        output_dir / "ellipses", geometry, clouds, planes, variances, residuals,
        ellipse_standard_deviations, dpi,
    )
    topic_sheets = _plot_topic_ellipse_sheets(
        output_dir / "topic_ellipses", records, geometry, clouds, planes, variances,
        ellipse_standard_deviations, dpi,
    )
    manifold = _plot_topic_manifold(
        output_dir / "topic_manifold.png", records, geometry,
        topic_ellipse_standard_deviations, dpi,
    )
    paper_gaussian_manifold = _plot_paper_gaussian_manifold(
        output_dir / "paper_gaussian_manifold.png", records, geometry,
        ellipse_standard_deviations, dpi,
    )
    mean_distances = pairwise_mean_squared(geometry.means)
    bures_distances = pairwise_model_bures(geometry)
    _plot_distance_matrices(
        output_dir / "topic_distances.png", records,
        mean_distances, bures_distances, dpi,
    )
    frame_items = _plot_frame_diagnostics(
        output_dir / "frames", records, geometry, dpi
    )
    frame_similarity = pairwise_frame_similarity(geometry.frames)
    _plot_frame_similarity(
        output_dir / "frame_subspace_similarity.png", records, frame_similarity, dpi
    )
    save_geometry(output_dir / "topic_geometry.npz", geometry)

    mean_metrics = cluster_metrics(mean_distances, topics)
    bures_metrics = cluster_metrics(bures_distances, topics)
    same_topic = np.equal.outer(topics, topics)
    off_diagonal = ~np.eye(len(topics), dtype=bool)
    within_frame = frame_similarity[same_topic & off_diagonal]
    between_frame = frame_similarity[~same_topic]
    report = {
        "dataset": {
            "name": dataset.get("dataset"),
            "retrieved_at": dataset.get("retrieved_at"),
            "source": dataset.get("source"),
            "paper_count": len(records),
            "topics": list(dict.fromkeys(topics)),
        },
        "method": {
            "manifold": (
                "PCA of learned means defines the display basis. The left panel zooms "
                "to mean geometry; the right preserves learned aggregate covariance scale."
            ),
            "topic_ellipses": (
                "Topic outlines are normalized products of projected learned paper "
                "Gaussians, using averaged precision and precision-weighted means."
            ),
            "paper_gaussian_manifold": (
                "Every point and ellipse is one abstract's own learned Gaussian in a "
                "shared mean-PCA plane; no topic aggregation is used."
            ),
            "ellipses": (
                "Every ellipse uses its paper's own top-two covariance directions; "
                "ellipse axes are never shared across papers."
            ),
            "frames": (
                "Frames are assessed with V^T V and intrinsic principal-angle "
                "similarity, without projecting frame directions into a common 2D plane."
            ),
            "claim_limit": (
                "PCA is illustrative. Full-space distance ratios and silhouettes are the "
                "evidence for or against topic bundling in this small sample."
            ),
        },
        "configuration": {
            "samples": samples,
            "ellipse_standard_deviations": ellipse_standard_deviations,
            "topic_ellipse_standard_deviations": topic_ellipse_standard_deviations,
            "seed": seed,
        },
        "cluster_metrics": {
            "learned_mean_squared_distance": mean_metrics,
            "model_bures_squared_distance": bures_metrics,
            "frame_subspace_similarity": {
                "within_topic_mean": float(within_frame.mean()),
                "between_topic_mean": float(between_frame.mean()),
                "within_minus_between": float(within_frame.mean() - between_frame.mean()),
            },
        },
        "manifold": manifold,
        "paper_gaussian_manifold": paper_gaussian_manifold,
        "papers": [
            {
                "arxiv_id": record["arxiv_id"],
                "topic": record["topic"],
                "title": record["title"],
                "url": record["url"],
                "mean_norm": float(np.linalg.norm(geometry.means[index])),
                "ellipse": ellipse_items[index],
                "frame": frame_items[index],
            }
            for index, record in enumerate(records)
        ],
        "artifacts": {
            "topic_manifold": "topic_manifold.png",
            "paper_gaussian_manifold": "paper_gaussian_manifold.png",
            "topic_distances": "topic_distances.png",
            "frame_subspace_similarity": "frame_subspace_similarity.png",
            "topic_ellipse_sheets": topic_sheets,
            "individual_ellipses": [item["file"] for item in ellipse_items],
            "individual_frames": [item["file"] for item in frame_items],
            "geometry": "topic_geometry.npz",
        },
    }
    (output_dir / "topic_analysis.json").write_text(json.dumps(report, indent=2))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--input-npz")
    source.add_argument("--demo", action="store_true")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default="visualizations/finetune/arxiv_topics")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--ellipse-std", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--topic-ellipse-std", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--gaussian-eps", type=float, default=1e-6)
    parser.add_argument("--decomposition-type", default="stiefel_projected_lambdas")
    parser.add_argument("--stiefel-iterations", type=int, default=4)
    parser.add_argument("--min-variance", type=float, default=1e-5)
    parser.add_argument("--max-variance", type=float, default=10.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataset, records = load_topic_records(args.dataset)
    if args.demo:
        geometry = synthetic_topic_geometry(records, seed=args.seed)
    elif args.input_npz:
        geometry = load_geometry(args.input_npz)
    else:
        geometry = checkpoint_topic_geometry(records, args)
    report = analyze_topic_geometry(
        dataset, records, geometry, args.output_dir,
        samples=args.samples,
        ellipse_standard_deviations=args.ellipse_std,
        seed=args.seed,
        topic_ellipse_standard_deviations=args.topic_ellipse_std,
        dpi=args.dpi,
    )
    print(f"Wrote arXiv topic analysis to {Path(args.output_dir).resolve()}")
    print(json.dumps(report["cluster_metrics"], indent=2))


if __name__ == "__main__":
    main()
