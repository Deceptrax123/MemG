"""LoCoMo query/reference/predicted-answer Gaussian visualizations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from finetune.visualize_gaussians import (
    covariance_in_basis,
    ellipse_points,
    orthogonality_error,
    shared_basis,
)


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def directional_statistics(reference_mean, direction, mean, frame, lambdas, diagonal):
    """Mean offset and variance along one common orthonormal direction."""
    reference_mean, direction, mean, frame, lambdas, diagonal = map(
        _numpy, (reference_mean, direction, mean, frame, lambdas, diagonal)
    )
    diagonal = np.diag(diagonal) if diagonal.ndim == 2 else diagonal
    mean_offset = float(direction @ (mean - reference_mean))
    low_rank = float(np.sum((direction @ frame) ** 2 * lambdas))
    residual = float(np.sum(direction**2 * diagonal))
    return mean_offset, max(low_rank + residual, 1e-12)


def normal_density(x, mean, variance):
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


class PlanningVisualizer:
    """Render LoCoMo Gaussian, dense-retrieval, and reranking diagnostics."""

    def __init__(self, output_dir="visualizations/locomo_planning", dpi=150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    @staticmethod
    def _matplotlib():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            return plt
        except ImportError as exc:
            raise RuntimeError("Install matplotlib to enable planning visualization") from exc

    def render_query_answer(
        self,
        *,
        question,
        reference_answer,
        predicted_answer,
        means,
        lambdas,
        frames,
        diagonals,
        plan=None,
        labels=None,
        texts=None,
    ):
        plt = self._matplotlib()
        labels = labels or ["query", "reference answer", "predicted answer"]
        texts = texts or [question, reference_answer, predicted_answer]
        means, lambdas, frames, diagonals = map(
            _numpy, (means, lambdas, frames, diagonals)
        )
        state_count = len(means)
        if state_count < 2 or len(labels) != state_count or len(texts) != state_count:
            raise ValueError("Gaussian visualization labels, texts, and distributions must match")

        basis = shared_basis(means, frames)
        centers = (means - means[0]) @ basis
        covariances = [
            covariance_in_basis(frames[i], lambdas[i], diagonals[i], basis)
            for i in range(state_count)
        ]
        output = self.output_dir / "gaussians.png"
        frame_output = self.output_dir / "gaussians_frames.png"
        color_map = plt.get_cmap("tab20")
        colors = [color_map(i / max(state_count - 1, 1)) for i in range(state_count)]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
        for i, label in enumerate(labels):
            ellipse = ellipse_points(centers[i], covariances[i])
            axes[0, 0].plot(ellipse[0], ellipse[1], color=colors[i], linewidth=2, label=label)
            axes[0, 0].scatter(*centers[i], color=colors[i], s=55, zorder=3)
        axes[0, 0].set_title("Queries, target entities, and answers (shared 2D projection, 2σ)")
        axes[0, 0].set_xlabel("shared component 1")
        axes[0, 0].set_ylabel("shared component 2")
        axes[0, 0].set_aspect("equal", adjustable="datalim")
        axes[0, 0].grid(alpha=0.2)
        axes[0, 0].legend()

        rank = lambdas.shape[1]
        x = np.arange(1, rank + 1)
        for i, label in enumerate(labels):
            axes[0, 1].plot(x, lambdas[i], marker="o", markersize=3, color=colors[i], label=label)
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("Low-rank eigenvalues by frame axis")
        axes[0, 1].set_xlabel("orthonormal axis k")
        axes[0, 1].set_ylabel("λₖ")
        axes[0, 1].set_xticks(x)
        axes[0, 1].grid(axis="y", alpha=0.2)
        axes[0, 1].legend()

        residuals = np.stack([np.diag(d) if d.ndim == 2 else d for d in diagonals])
        axes[1, 0].boxplot(
            [np.log10(np.clip(row, 1e-12, None)) for row in residuals],
            tick_labels=labels,
            patch_artist=True,
        )
        axes[1, 0].set_title("Residual diagonal variance across embedding dimensions")
        axes[1, 0].set_ylabel("log₁₀ diagonal variance")
        axes[1, 0].tick_params(axis="x", rotation=35)
        axes[1, 0].grid(axis="y", alpha=0.2)

        alignments = [np.abs(frames[0].T @ frames[i]) for i in range(1, state_count)]
        combined = np.concatenate(alignments, axis=1)
        image = axes[1, 1].imshow(combined, vmin=0, vmax=1, cmap="magma", aspect="auto")
        for boundary in range(rank, combined.shape[1], rank):
            axes[1, 1].axvline(boundary - 0.5, color="white", linewidth=1.5)
        axes[1, 1].set_title(f"|V{labels[0]}ᵀ Vstate| axis alignment")
        axes[1, 1].set_xlabel("comparison state and frame axis")
        axes[1, 1].set_ylabel(f"{labels[0]} frame axis")
        axes[1, 1].set_yticks(range(rank), [f"q{k + 1}" for k in range(rank)])
        x_labels = [
            f"{labels[state][:10]}:{axis + 1}"
            for state in range(1, state_count)
            for axis in range(rank)
        ]
        axes[1, 1].set_xticks(
            range(len(x_labels)), x_labels, rotation=60, ha="right", fontsize=7,
        )
        fig.colorbar(image, ax=axes[1, 1], fraction=0.046)

        difficulty = (plan or {}).get("difficulty_level", "unknown")
        fig.suptitle(f"{difficulty}: {question}", fontsize=12)
        fig.savefig(output, dpi=self.dpi)
        plt.close(fig)

        directional = self._render_frame_report(
            frame_output,
            labels=labels,
            colors=colors,
            means=means,
            lambdas=lambdas,
            frames=frames,
            diagonals=diagonals,
            question=question,
        )

        metadata = {
            "question": question,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
            "plan": plan or {},
            "labels": labels,
            "texts": [str(text) for text in texts],
            "projection_note": "Ellipses are shared 2D projections of high-dimensional Gaussians.",
            "frame_note": (
                f"V is one d×{rank} Stiefel frame containing {rank} orthonormal axes; "
                "the frame report plots one common query-frame coordinate per axis."
            ),
            "states": [],
            "reference_frame_label": labels[0],
            "frame_alignments": {
                labels[i]: alignments[i - 1].tolist() for i in range(1, state_count)
            },
            "directional_distributions": directional,
        }
        for i, label in enumerate(labels):
            error, gram = orthogonality_error(frames[i])
            metadata["states"].append({
                "label": label,
                "text": str(texts[i]),
                "mean_norm": float(np.linalg.norm(means[i])),
                "lambda_min": float(lambdas[i].min()),
                "lambda_mean": float(lambdas[i].mean()),
                "lambda_max": float(lambdas[i].max()),
                "orthogonality_error_fro": error,
                "gram_matrix": gram.tolist(),
            })
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))
        return output

    def render_retrieval_hits(
        self, *, question, query_labels, records, max_candidates=30
    ):
        """Plot the exact Wasserstein-proxy distances returned by FAISS."""
        if not records:
            return None
        plt = self._matplotlib()
        output = self.output_dir / "retrieval_wasserstein.png"
        displayed = [record for record in records if record["hit_rank"] <= max_candidates]
        label_order = [label for label in query_labels if any(
            record["query_label"] == label for record in displayed
        )]
        color_map = plt.get_cmap("tab10")

        fig, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)
        for index, label in enumerate(label_order):
            hits = sorted(
                (record for record in displayed if record["query_label"] == label),
                key=lambda record: record["hit_rank"],
            )
            axes[0, 0].plot(
                [record["hit_rank"] for record in hits],
                [max(record["wasserstein_squared"], 1e-12) for record in hits],
                marker="o", markersize=3, linewidth=1.5,
                color=color_map(index % 10), label=label,
            )
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("FAISS hit distance by rank")
        axes[0, 0].set_xlabel("FAISS hit rank")
        axes[0, 0].set_ylabel("diagonal Gaussian W₂² proxy")
        axes[0, 0].grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)

        best_by_node = {}
        for record in displayed:
            current = best_by_node.get(record["node_id"])
            if current is None or record["wasserstein_squared"] < current["wasserstein_squared"]:
                best_by_node[record["node_id"]] = record
        node_ids = [
            record["node_id"] for record in sorted(
                best_by_node.values(), key=lambda record: record["wasserstein_squared"]
            )[:max_candidates]
        ]
        heatmap = np.full((len(label_order), len(node_ids)), np.nan)
        lookup = {(record["query_label"], record["node_id"]): record for record in displayed}
        for row, label in enumerate(label_order):
            for column, node_id in enumerate(node_ids):
                record = lookup.get((label, node_id))
                if record is not None:
                    heatmap[row, column] = np.log1p(record["wasserstein_squared"])
        image = axes[0, 1].imshow(
            np.ma.masked_invalid(heatmap), cmap="viridis", aspect="auto"
        )
        axes[0, 1].set_title("Query-to-hit distance matrix")
        axes[0, 1].set_xlabel("memory hit (best-distance order)")
        axes[0, 1].set_ylabel("query variant")
        axes[0, 1].set_xticks(range(len(node_ids)), node_ids, rotation=65, ha="right", fontsize=7)
        axes[0, 1].set_yticks(range(len(label_order)), label_order)
        fig.colorbar(image, ax=axes[0, 1], label="log1p W₂² proxy")

        best_hits = [best_by_node[node_id] for node_id in node_ids]
        x = np.arange(len(best_hits))
        width = 0.35
        axes[1, 0].bar(
            x - width / 2,
            np.log1p([record["mean_energy"] for record in best_hits]),
            width, color="#2563eb", label="log1p mean contribution",
        )
        axes[1, 0].bar(
            x + width / 2,
            np.log1p([record["std_profile_energy"] for record in best_hits]),
            width, color="#a855f7", label="log1p std-profile contribution",
        )
        axes[1, 0].plot(
            x, np.log1p([record["wasserstein_squared"] for record in best_hits]),
            color="#111827", marker="D", markersize=3, linewidth=1, label="log1p total",
        )
        axes[1, 0].set_xticks(x, node_ids, rotation=65, ha="right", fontsize=7)
        axes[1, 0].set_title("W₂² components for the strongest unique hits")
        axes[1, 0].set_ylabel("log1p distance contribution")
        axes[1, 0].grid(axis="y", alpha=0.2)
        axes[1, 0].legend(fontsize=8)

        consistency = []
        for node_id in node_ids:
            node_records = [record for record in displayed if record["node_id"] == node_id]
            consistency.append((
                node_id,
                len(node_records),
                sum(1.0 / record["hit_rank"] for record in node_records),
            ))
        consistency.sort(key=lambda item: (-item[2], -item[1], item[0]))
        cx = np.arange(len(consistency))
        axes[1, 1].bar(
            cx, [item[2] for item in consistency], color="#0f766e",
            label="sum reciprocal rank",
        )
        axes[1, 1].plot(
            cx, [item[1] for item in consistency], color="#f59e0b",
            marker="o", linewidth=1.5, label="query-variant hit count",
        )
        axes[1, 1].set_xticks(
            cx, [item[0] for item in consistency], rotation=65, ha="right", fontsize=7
        )
        axes[1, 1].set_title("Hit consistency across query variants")
        axes[1, 1].set_ylabel("consistency score / count")
        axes[1, 1].grid(axis="y", alpha=0.2)
        axes[1, 1].legend(fontsize=8)

        fig.suptitle(f"Wasserstein-based FAISS retrieval hits: {question}", fontsize=13)
        fig.savefig(output, dpi=self.dpi)
        plt.close(fig)
        payload = {
            "question": question,
            "query_labels": query_labels,
            "captured_hit_count": len(records),
            "displayed_hit_count": len(displayed),
            "formula": "W2_proxy^2 = ||mu_q-mu_n||^2 + ||sqrt(diag Sigma_q)-sqrt(diag Sigma_n)||^2",
            "faiss_note": "These are the actual IndexFlatL2 outputs over [mu, sqrt(diag Sigma)].",
            "display_note": f"The PNG shows at most {max_candidates} hits per query variant.",
            "records": records,
        }
        output.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
        return output

    def render_reranking(self, *, question, records, max_candidates=30):
        """Plot energy decomposition and rank movement through both stages."""
        if not records:
            return None
        plt = self._matplotlib()
        output = self.output_dir / "reranking.png"
        ordered = sorted(records, key=lambda item: item["stage1_rank"])
        displayed = ordered[:max_candidates]
        labels = [
            ("★" if item["selected"] else "") + str(item["node_id"])
            for item in displayed
        ]
        selected = np.array([bool(item["selected"]) for item in displayed])
        colors = np.where(selected, "#16a34a", "#94a3b8")

        fig, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)
        x = np.arange(len(displayed))
        stage1_mean = np.log1p([item["stage1_mean_energy"] for item in displayed])
        stage1_cov = np.log1p([item["stage1_covariance_energy"] for item in displayed])
        width = 0.36
        axes[0, 0].bar(x - width / 2, stage1_mean, width, color="#2563eb", label="log1p mean")
        axes[0, 0].bar(x + width / 2, stage1_cov, width, color="#f59e0b", label="log1p covariance")
        axes[0, 0].plot(
            x, np.log1p([item["stage1_total_energy"] for item in displayed]),
            color="#111827", marker="D", markersize=3, linewidth=1, label="log1p total",
        )
        axes[0, 0].set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
        axes[0, 0].set_title("Stage 1 geometric-energy components (stage-1 order)")
        axes[0, 0].set_ylabel("log1p energy")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(axis="y", alpha=0.2)

        stage2_records = sorted(
            [item for item in records if "stage2_rank" in item],
            key=lambda item: item["stage2_rank"],
        )[:max_candidates]
        x2 = np.arange(len(stage2_records))
        labels2 = [
            ("★" if item["selected"] else "") + str(item["node_id"])
            for item in stage2_records
        ]
        mean2 = np.log1p([item["stage2_mean_energy"] for item in stage2_records])
        variance2 = np.log1p([item["stage2_variance_energy"] for item in stage2_records])
        axes[0, 1].bar(x2 - width / 2, mean2, width, color="#2563eb", label="log1p mean")
        axes[0, 1].bar(x2 + width / 2, variance2, width, color="#a855f7", label="log1p std-profile")
        axes[0, 1].plot(
            x2, np.log1p([item["stage2_total_energy"] for item in stage2_records]),
            color="#111827", marker="D", markersize=3, linewidth=1, label="log1p total",
        )
        axes[0, 1].set_xticks(x2, labels2, rotation=70, ha="right", fontsize=7)
        axes[0, 1].set_title("Stage 2 reranker components (reranker order)")
        axes[0, 1].set_ylabel("log1p energy")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(axis="y", alpha=0.2)

        for item, color in zip(displayed, colors):
            ranks = [item["stage1_rank"]]
            positions = [0]
            if "stage2_rank" in item:
                ranks.append(item["stage2_rank"])
                positions.append(1)
            if item.get("chronological_output_position") is not None:
                ranks.append(item["chronological_output_position"])
                positions.append(2)
            axes[1, 0].plot(
                positions, ranks, marker="o", color=color,
                alpha=0.9 if item["selected"] else 0.25,
                linewidth=2 if item["selected"] else 0.8,
            )
            if item["selected"]:
                axes[1, 0].annotate(
                    str(item["node_id"]), (positions[-1], ranks[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=7,
                )
        axes[1, 0].set_xticks(
            [0, 1, 2], ["stage 1 rank", "stage 2 rank", "chronological output"]
        )
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_ylabel("rank / output position (1 is highest)")
        axes[1, 0].set_title("Candidate rank movement; selected memories are green")
        axes[1, 0].grid(alpha=0.2)

        comparable = [item for item in displayed if "stage2_total_energy" in item]
        scatter_colors = ["#16a34a" if item["selected"] else "#64748b" for item in comparable]
        axes[1, 1].scatter(
            [np.log1p(item["stage1_total_energy"]) for item in comparable],
            [np.log1p(item["stage2_total_energy"]) for item in comparable],
            c=scatter_colors,
            s=[55 if item["selected"] else 24 for item in comparable],
            alpha=0.8,
        )
        for item in comparable:
            if item["selected"]:
                axes[1, 1].annotate(
                    str(item["node_id"]),
                    (np.log1p(item["stage1_total_energy"]), np.log1p(item["stage2_total_energy"])),
                    fontsize=7, xytext=(3, 3), textcoords="offset points",
                )
        axes[1, 1].set_xlabel("log1p stage-1 geometric energy")
        axes[1, 1].set_ylabel("log1p stage-2 reranker energy")
        axes[1, 1].set_title("Agreement between scoring stages")
        axes[1, 1].grid(alpha=0.2)

        fig.suptitle(f"Retrieval energy and reranking: {question}", fontsize=13)
        fig.savefig(output, dpi=self.dpi)
        plt.close(fig)
        payload = {
            "question": question,
            "candidate_count": len(records),
            "displayed_candidate_count": len(displayed),
            "stage1_formula": "||mu_q-mu_n||^2 + 0.5||Sigma_q-Sigma_n||_F^2",
            "stage2_formula": "||mu_q-mu_n||^2 + ||sqrt(diag Sigma_q)-sqrt(diag Sigma_n)||^2",
            "ordering_note": "Selected stage-2 results are sorted chronologically before answer generation.",
            "lambda_note": "The current agent scorers use V V^T and do not apply learned lambda values.",
            "records": records,
        }
        output.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
        return output

    def _render_frame_report(
        self, output, *, labels, colors, means, lambdas, frames, diagonals, question
    ):
        plt = self._matplotlib()
        rank, dimension = frames.shape[2], frames.shape[1]
        columns = min(4, rank)
        rows = math.ceil(rank / columns)
        state_count = len(labels)
        fig = plt.figure(
            figsize=(18, 2.7 * state_count + 3.5 * rows), constrained_layout=True
        )
        grid = fig.add_gridspec(state_count + rows, columns)

        max_loading = max(float(np.abs(frame.T).max()) for frame in frames) or 1.0
        for state_index, (label, color) in enumerate(zip(labels, colors)):
            axis = fig.add_subplot(grid[state_index, :])
            image = axis.imshow(
                frames[state_index].T,
                cmap="coolwarm",
                vmin=-max_loading,
                vmax=max_loading,
                aspect="auto",
                interpolation="nearest",
            )
            axis.set_title(f"{label}: Stiefel loading matrix Vᵀ ({rank} axes × {dimension} dimensions)")
            axis.set_ylabel("axis k")
            axis.set_yticks(range(rank), [str(k + 1) for k in range(rank)])
            axis.set_xlabel("embedding dimension")
            fig.colorbar(image, ax=axis, fraction=0.012, pad=0.01)

        directional = []
        query_frame = frames[0]
        for k in range(rank):
            axis = fig.add_subplot(grid[state_count + k // columns, k % columns])
            direction = query_frame[:, k]
            stats = [
                directional_statistics(means[0], direction, means[i], frames[i], lambdas[i], diagonals[i])
                for i in range(state_count)
            ]
            lower = min(mean - 4 * np.sqrt(var) for mean, var in stats)
            upper = max(mean + 4 * np.sqrt(var) for mean, var in stats)
            x = np.linspace(lower, upper, 320)
            record = {"query_axis": k + 1, "states": []}
            for i, (label, color) in enumerate(zip(labels, colors)):
                mean, variance = stats[i]
                axis.plot(x, normal_density(x, mean, variance), color=color, linewidth=2, label=label)
                axis.axvline(mean, color=color, alpha=0.35, linewidth=1)
                record["states"].append({
                    "label": label,
                    "mean_offset": mean,
                    "variance": variance,
                    "standard_deviation": float(np.sqrt(variance)),
                })
            axis.set_title(f"Query-frame axis {k + 1} (λq={lambdas[0, k]:.3g})")
            axis.set_xlabel("coordinate relative to query mean")
            axis.set_ylabel("density")
            axis.grid(alpha=0.2)
            if k == 0:
                axis.legend(fontsize=8)
            directional.append(record)

        for index in range(rank, rows * columns):
            axis = fig.add_subplot(grid[state_count + index // columns, index % columns])
            axis.axis("off")
        fig.suptitle(f"Low-rank frame distributions: {question}", fontsize=13)
        fig.savefig(output, dpi=self.dpi)
        plt.close(fig)
        return directional
