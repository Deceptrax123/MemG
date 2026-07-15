"""Visualize learned Gaussian retrieval geometry on local MS MARCO samples."""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from finetune.offline_z_analysis import (
    Geometry,
    _array,
    _matplotlib,
    _plot_learned_ellipses,
    learned_local_ellipse_planes,
    sample_gaussians,
    save_geometry,
)
from finetune.visualize_arxiv_topics import (
    _model_and_tokenizer,
    _project_paper_covariances,
    pairwise_mean_squared,
    pairwise_model_bures,
)


ROLE_COLORS = {
    "query": "#4c78a8",
    "positive": "#54a24b",
    "negative": "#e45756",
}
ROLE_MARKERS = {"query": "o", "positive": "s", "negative": "X"}
ROLE_LINESTYLES = {"query": "-", "positive": "--", "negative": ":"}


def sample_msmarco_groups(
    shard_paths: list[str | Path],
    *,
    groups: int,
    negatives_per_query: int,
    scan_records: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Reservoir-sample valid query/positive/hard-negative groups from JSONL."""
    if groups < 1 or negatives_per_query < 1 or scan_records < groups:
        raise ValueError("groups/negatives must be positive and scan_records >= groups")
    paths = [Path(path) for path in shard_paths]
    if not paths:
        raise FileNotFoundError("No MS MARCO JSONL shards matched")
    rng = random.Random(seed)
    rng.shuffle(paths)
    reservoir: list[dict[str, Any]] = []
    valid_seen = 0
    scanned = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if scanned >= scan_records:
                    break
                scanned += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query = str(record.get("query", "")).strip()
                positive = str(record.get("positive", "")).strip()
                negatives = [
                    str(text).strip()
                    for text in record.get("negatives", [])
                    if str(text).strip()
                ]
                negatives = list(dict.fromkeys(
                    text for text in negatives if text != positive
                ))
                if not query or not positive or len(negatives) < negatives_per_query:
                    continue
                chosen_negatives = rng.sample(negatives, negatives_per_query)
                candidate = {
                    "query": query,
                    "positive": positive,
                    "negatives": chosen_negatives,
                    "source_file": str(path),
                    "source_line": line_number,
                }
                valid_seen += 1
                if len(reservoir) < groups:
                    reservoir.append(candidate)
                else:
                    replacement = rng.randrange(valid_seen)
                    if replacement < groups:
                        reservoir[replacement] = candidate
            if scanned >= scan_records:
                break
    if len(reservoir) < groups:
        raise ValueError(
            f"Only found {len(reservoir)} valid groups while scanning {scanned} records"
        )
    return reservoir


def flatten_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten groups while preserving retrieval role and query-group identity."""
    records = []
    for group_index, group in enumerate(groups):
        group_id = f"q{group_index + 1:02d}"
        records.append({
            "label": f"{group_id}_query",
            "short_label": f"Q{group_index + 1}",
            "group_index": group_index,
            "group_id": group_id,
            "role": "query",
            "text": group["query"],
        })
        records.append({
            "label": f"{group_id}_positive",
            "short_label": f"P{group_index + 1}",
            "group_index": group_index,
            "group_id": group_id,
            "role": "positive",
            "text": group["positive"],
        })
        for negative_index, text in enumerate(group["negatives"], start=1):
            records.append({
                "label": f"{group_id}_negative_{negative_index}",
                "short_label": f"N{group_index + 1}.{negative_index}",
                "group_index": group_index,
                "group_id": group_id,
                "role": "negative",
                "negative_index": negative_index,
                "text": text,
            })
    return records


def checkpoint_geometry(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> Geometry:
    """Encode sampled MS MARCO text using a finetuned Gaussian checkpoint."""
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
                [record["text"] for record in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
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
        labels=[record["label"] for record in records],
        texts=[record["text"] for record in records],
        positive_indices=[
            index for index, record in enumerate(records)
            if record["role"] == "positive"
        ],
        z=np.concatenate(collected["z"]),
        means=np.concatenate(collected["means"]),
        lambdas=np.concatenate(collected["lambdas"]),
        frames=np.concatenate(collected["frames"]),
        diagonals=np.concatenate(collected["diagonals"]),
    ).validate()


def _mean_pca(geometry: Geometry) -> tuple[np.ndarray, np.ndarray, list[float]]:
    centered = geometry.means - geometry.means.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    projected = centered @ basis
    total = float(np.sum(singular_values ** 2))
    explained = (
        singular_values[:2] ** 2 / max(total, 1e-12)
    ).tolist()
    return basis, projected, explained


def _ellipse_contour(
    center: np.ndarray,
    covariance: np.ndarray,
    standard_deviations: float,
) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    values = np.clip(values, 1e-12, None)
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    axes = standard_deviations * np.sqrt(values)
    return center[:, None] + vectors @ np.diag(axes) @ circle, axes


def plot_shared_manifold(
    output: Path,
    records: list[dict[str, Any]],
    geometry: Geometry,
    *,
    standard_deviations: float,
    dpi: int,
) -> dict[str, Any]:
    """Plot every sampled item with its own covariance; no aggregation."""
    plt = _matplotlib()
    basis, projected, explained = _mean_pca(geometry)
    covariances = _project_paper_covariances(geometry, basis)
    group_count = 1 + max(record["group_index"] for record in records)
    group_colors = plt.cm.tab10(np.linspace(0.0, 1.0, group_count))
    figure, axis = plt.subplots(figsize=(13, 10), constrained_layout=True)
    seen_roles = set()
    items = []
    for index, record in enumerate(records):
        contour, ellipse_axes = _ellipse_contour(
            projected[index], covariances[index], standard_deviations
        )
        role = record["role"]
        color = group_colors[record["group_index"]]
        axis.plot(
            contour[0], contour[1], color=color,
            linestyle=ROLE_LINESTYLES[role], linewidth=1.15, alpha=0.48,
        )
        role_label = role.title() if role not in seen_roles else None
        seen_roles.add(role)
        axis.scatter(
            *projected[index], color=ROLE_COLORS[role],
            marker=ROLE_MARKERS[role], s=58, edgecolor="black",
            linewidth=0.45, label=role_label, zorder=3,
        )
        axis.annotate(
            record["short_label"], projected[index],
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
        items.append({
            "label": record["label"],
            "role": role,
            "group_id": record["group_id"],
            "projected_mean": projected[index].tolist(),
            "projected_covariance": covariances[index].tolist(),
            "ellipse_axis_lengths": ellipse_axes.tolist(),
        })
    axis.set(
        title="MS MARCO learned Gaussians — every ellipse belongs to one item",
        xlabel=f"mean PCA 1 ({explained[0] * 100:.1f}% variance)",
        ylabel=f"mean PCA 2 ({explained[1] * 100:.1f}% variance)",
        aspect="equal",
    )
    axis.grid(alpha=0.2)
    axis.legend(title="Retrieval role")
    axis.text(
        0.01, 0.01,
        "Line color identifies query group; point marker/color identifies role. "
        "No group covariance aggregation.",
        transform=axis.transAxes, fontsize=9, alpha=0.8,
    )
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return {
        "explained_variance_ratio": explained,
        "aggregation": "none",
        "items": items,
    }


def plot_group_panels(
    output: Path,
    records: list[dict[str, Any]],
    geometry: Geometry,
    *,
    standard_deviations: float,
    dpi: int,
) -> None:
    """Plot one query/positive/negative ellipse panel per retrieval group."""
    plt = _matplotlib()
    basis, projected, _ = _mean_pca(geometry)
    covariances = _project_paper_covariances(geometry, basis)
    group_count = 1 + max(record["group_index"] for record in records)
    columns = min(3, group_count)
    rows = math.ceil(group_count / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(7 * columns, 6 * rows),
        sharex=True, sharey=True, constrained_layout=True, squeeze=False,
    )
    for group_index, axis in enumerate(axes.flat[:group_count]):
        indices = [
            index for index, record in enumerate(records)
            if record["group_index"] == group_index
        ]
        for index in indices:
            record = records[index]
            role = record["role"]
            contour, _ = _ellipse_contour(
                projected[index], covariances[index], standard_deviations
            )
            axis.plot(
                contour[0], contour[1], color=ROLE_COLORS[role],
                linestyle=ROLE_LINESTYLES[role], linewidth=1.8, alpha=0.85,
            )
            axis.scatter(
                *projected[index], color=ROLE_COLORS[role],
                marker=ROLE_MARKERS[role], edgecolor="black",
                linewidth=0.5, s=64, zorder=3,
            )
            axis.annotate(
                record["short_label"], projected[index],
                xytext=(5, 5), textcoords="offset points", fontsize=8,
            )
        axis.set(
            title=f"Query group {group_index + 1}",
            xlabel="mean PCA 1",
            ylabel="mean PCA 2",
            aspect="equal",
        )
        axis.grid(alpha=0.2)
    for axis in axes.flat[group_count:]:
        axis.set_visible(False)
    figure.suptitle(
        "MS MARCO retrieval groups: query, positive, and hard-negative "
        f"paper-specific Gaussians ({standard_deviations:g}σ)"
    )
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def retrieval_records(
    records: list[dict[str, Any]],
    mean_distances: np.ndarray,
    bures_distances: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compare each positive with its same-query sampled hard negatives."""
    output = []
    group_count = 1 + max(record["group_index"] for record in records)
    for group_index in range(group_count):
        query = next(
            index for index, record in enumerate(records)
            if record["group_index"] == group_index and record["role"] == "query"
        )
        candidates = [
            index for index, record in enumerate(records)
            if record["group_index"] == group_index and record["role"] != "query"
        ]
        for index in candidates:
            output.append({
                "group_id": records[index]["group_id"],
                "candidate_label": records[index]["label"],
                "role": records[index]["role"],
                "mean_squared_distance": float(mean_distances[query, index]),
                "bures_squared_distance": float(bures_distances[query, index]),
            })
    for metric in ("mean_squared_distance", "bures_squared_distance"):
        for group_index in range(group_count):
            group_id = f"q{group_index + 1:02d}"
            group_items = [item for item in output if item["group_id"] == group_id]
            ordered = sorted(group_items, key=lambda item: item[metric])
            for rank, item in enumerate(ordered, start=1):
                item[f"{metric}_rank"] = rank
    positives = [item for item in output if item["role"] == "positive"]
    negatives = [item for item in output if item["role"] == "negative"]
    metrics = {
        "mean_positive_top1_rate": float(np.mean([
            item["mean_squared_distance_rank"] == 1 for item in positives
        ])),
        "bures_positive_top1_rate": float(np.mean([
            item["bures_squared_distance_rank"] == 1 for item in positives
        ])),
        "mean_positive_distance": float(np.mean([
            item["mean_squared_distance"] for item in positives
        ])),
        "mean_negative_distance": float(np.mean([
            item["mean_squared_distance"] for item in negatives
        ])),
        "bures_positive_distance": float(np.mean([
            item["bures_squared_distance"] for item in positives
        ])),
        "bures_negative_distance": float(np.mean([
            item["bures_squared_distance"] for item in negatives
        ])),
    }
    return output, metrics


def plot_retrieval_distances(
    output: Path, retrieval: list[dict[str, Any]], *, dpi: int
) -> None:
    plt = _matplotlib()
    groups = list(dict.fromkeys(item["group_id"] for item in retrieval))
    figure, axes = plt.subplots(
        1, 2, figsize=(16, 6), constrained_layout=True
    )
    for axis, metric, title in (
        (axes[0], "mean_squared_distance", "Learned-mean squared distance"),
        (axes[1], "bures_squared_distance", "Covariance-aware squared distance"),
    ):
        width = 0.8 / max(len([
            item for item in retrieval if item["group_id"] == groups[0]
        ]), 1)
        for group_position, group_id in enumerate(groups):
            items = [item for item in retrieval if item["group_id"] == group_id]
            for candidate_position, item in enumerate(items):
                x = group_position - 0.4 + width / 2 + candidate_position * width
                axis.bar(
                    x, item[metric], width=width,
                    color=ROLE_COLORS[item["role"]],
                    alpha=0.85,
                )
        axis.set_xticks(np.arange(len(groups)), groups)
        axis.set(title=title, xlabel="query group", ylabel="distance (lower is better)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Green is the positive passage; red bars are hard negatives")
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def analyze(
    output_dir: str | Path,
    groups: list[dict[str, Any]],
    records: list[dict[str, Any]],
    geometry: Geometry,
    *,
    checkpoint: str,
    standard_deviations: float,
    samples: int,
    seed: int,
    dpi: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = geometry.validate()
    if geometry.labels != [record["label"] for record in records]:
        raise ValueError("Geometry labels must match flattened MS MARCO records")
    rng = np.random.default_rng(seed)
    clouds = sample_gaussians(geometry, samples, rng)
    planes, variances, residuals = learned_local_ellipse_planes(geometry)
    ellipse_items = _plot_learned_ellipses(
        output_dir / "ellipses", geometry, clouds, planes, variances, residuals,
        standard_deviations, dpi,
    )
    manifold = plot_shared_manifold(
        output_dir / "msmarco_gaussian_manifold.png",
        records, geometry, standard_deviations=standard_deviations, dpi=dpi,
    )
    plot_group_panels(
        output_dir / "msmarco_query_groups.png",
        records, geometry, standard_deviations=standard_deviations, dpi=dpi,
    )
    mean_distances = pairwise_mean_squared(geometry.means)
    bures_distances = pairwise_model_bures(geometry)
    retrieval, metrics = retrieval_records(records, mean_distances, bures_distances)
    plot_retrieval_distances(
        output_dir / "msmarco_retrieval_distances.png", retrieval, dpi=dpi
    )
    save_geometry(output_dir / "msmarco_geometry.npz", geometry)
    (output_dir / "msmarco_samples.json").write_text(json.dumps({
        "source": "local MS MARCO hard-negative training shards",
        "groups": groups,
    }, indent=2))
    report = {
        "dataset": {
            "name": "MS MARCO hard-negative training samples",
            "group_count": len(groups),
            "item_count": len(records),
            "roles": {
                role: sum(record["role"] == role for record in records)
                for role in ROLE_COLORS
            },
        },
        "model": {
            "checkpoint": checkpoint,
            "output_dimension": int(geometry.means.shape[1]),
            "rank": int(geometry.lambdas.shape[1]),
            "diagonal_min": float(geometry.diagonals.min()),
            "diagonal_max": float(geometry.diagonals.max()),
        },
        "method": {
            "ellipses": (
                "Every outline is the covariance of that query or passage centered "
                "on its own learned mean; no group covariance aggregation is used."
            ),
            "retrieval_scope": (
                "Each positive is ranked only against the sampled hard negatives "
                "belonging to the same query group."
            ),
        },
        "configuration": {
            "standard_deviations": standard_deviations,
            "samples": samples,
            "seed": seed,
        },
        "retrieval_metrics": metrics,
        "retrieval_records": retrieval,
        "manifold": manifold,
        "individual_ellipses": ellipse_items,
        "artifacts": {
            "shared_manifold": "msmarco_gaussian_manifold.png",
            "query_group_panels": "msmarco_query_groups.png",
            "retrieval_distances": "msmarco_retrieval_distances.png",
            "samples": "msmarco_samples.json",
            "geometry": "msmarco_geometry.npz",
            "individual_ellipses": [item["file"] for item in ellipse_items],
        },
    }
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--shards", default="data_binaries/negatives_part_*.jsonl"
    )
    parser.add_argument(
        "--output-dir", default="visualizations/finetune/msmarco_samples"
    )
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--negatives-per-query", type=int, default=2)
    parser.add_argument("--scan-records", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--ellipse-std", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--gaussian-eps", type=float, default=1e-6)
    parser.add_argument(
        "--decomposition-type", default="stiefel_projected_lambdas"
    )
    parser.add_argument("--stiefel-iterations", type=int, default=4)
    parser.add_argument("--min-variance", type=float, default=1e-5)
    parser.add_argument("--max-variance", type=float, default=10.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    shard_paths = sorted(glob.glob(args.shards))
    groups = sample_msmarco_groups(
        shard_paths,
        groups=args.groups,
        negatives_per_query=args.negatives_per_query,
        scan_records=args.scan_records,
        seed=args.seed,
    )
    records = flatten_groups(groups)
    geometry = checkpoint_geometry(records, args)
    report = analyze(
        args.output_dir,
        groups,
        records,
        geometry,
        checkpoint=args.checkpoint,
        standard_deviations=args.ellipse_std,
        samples=args.samples,
        seed=args.seed,
        dpi=args.dpi,
    )
    print(f"Wrote MS MARCO analysis to {Path(args.output_dir).resolve()}")
    print(json.dumps(report["retrieval_metrics"], indent=2))


if __name__ == "__main__":
    main()
