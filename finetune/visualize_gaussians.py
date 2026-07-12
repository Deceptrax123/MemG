"""Visual diagnostics for Gaussian query/passage embeddings.

Run standalone with a checkpoint, or call ``visualize_projection_batch`` from
training. The plots use a shared two-dimensional PCA-like basis constructed
from means and Stiefel directions; projected ellipses are diagnostics, not the
full high-dimensional distributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from finetune.model import GaussianEmbeddingTuner
from transformers import AutoTokenizer, BertTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def covariance_in_basis(V, lambdas, D, basis):
    """Project D + V diag(lambda) V^T without materializing the full covariance."""
    V, lambdas, D, basis = map(_numpy, (V, lambdas, D, basis))
    diagonal = np.diag(D) if D.ndim == 2 else D
    projected_frame = basis.T @ V
    return (
        basis.T @ (diagonal[:, None] * basis)
        + (projected_frame * lambdas[None, :]) @ projected_frame.T
    )


def shared_basis(means, frames):
    means = _numpy(means)
    frames = _numpy(frames)
    centered = means - means.mean(axis=0, keepdims=True)
    candidates = [centered]
    for frame in frames:
        candidates.append(frame.T)
    matrix = np.concatenate(candidates, axis=0)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    basis = vt[:2].T
    if basis.shape[1] < 2:
        eye = np.eye(means.shape[1])
        basis = np.column_stack([basis, eye[:, : 2 - basis.shape[1]]])
    basis, _ = np.linalg.qr(basis)
    return basis[:, :2]


def orthogonality_error(frame):
    frame = _numpy(frame)
    gram = frame.T @ frame
    return float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro")), gram


def ellipse_points(center, covariance, standard_deviations=2.0, points=180):
    covariance = (_numpy(covariance) + _numpy(covariance).T) / 2
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 1e-12, None)
    circle = np.stack([
        np.cos(np.linspace(0, 2 * np.pi, points)),
        np.sin(np.linspace(0, 2 * np.pi, points)),
    ])
    transform = vectors @ np.diag(np.sqrt(values) * standard_deviations)
    return _numpy(center)[:, None] + transform @ circle


def visualize_projection_batch(
    output_path,
    *,
    labels,
    means,
    lambdas,
    frames,
    diagonal_matrices,
    title="Gaussian embedding diagnostics",
    dpi=150,
):
    """Write a four-panel PNG and adjacent JSON diagnostics file."""
    

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means, lambdas, frames, diagonal_matrices = map(
        _numpy, (means, lambdas, frames, diagonal_matrices)
    )
    count = min(len(labels), len(means), len(lambdas), len(frames), len(diagonal_matrices))
    labels = list(labels[:count])
    means, lambdas = means[:count], lambdas[:count]
    frames, diagonal_matrices = frames[:count], diagonal_matrices[:count]
    basis = shared_basis(means, frames)
    projected_means = means @ basis

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(count, 1)))
    diagnostics = {"title": title, "items": []}

    for index, label in enumerate(labels):
        covariance_2d = covariance_in_basis(
            frames[index], lambdas[index], diagonal_matrices[index], basis
        )
        ellipse = ellipse_points(projected_means[index], covariance_2d)
        axes[0, 0].plot(ellipse[0], ellipse[1], color=colors[index], linewidth=2)
        axes[0, 0].scatter(*projected_means[index], color=colors[index], s=45)
        axes[0, 0].annotate(label, projected_means[index], fontsize=8)

        axes[0, 1].plot(
            np.arange(1, len(lambdas[index]) + 1),
            np.sort(lambdas[index])[::-1],
            marker="o", markersize=3, label=label, color=colors[index],
        )

        projected_frame = basis.T @ frames[index]
        scale = np.sqrt(np.clip(lambdas[index], 1e-12, None))
        for rank_index in range(min(projected_frame.shape[1], 8)):
            arrow = projected_frame[:, rank_index] * scale[rank_index]
            axes[1, 0].arrow(
                projected_means[index, 0], projected_means[index, 1],
                arrow[0], arrow[1], color=colors[index], alpha=0.45,
                head_width=0.03, length_includes_head=True,
            )

        error, gram = orthogonality_error(frames[index])
        diagnostics["items"].append({
            "label": label,
            "mean_norm": float(np.linalg.norm(means[index])),
            "lambda_min": float(lambdas[index].min()),
            "lambda_mean": float(lambdas[index].mean()),
            "lambda_max": float(lambdas[index].max()),
            "diagonal_variance_mean": float(np.diag(diagonal_matrices[index]).mean()),
            "orthogonality_error_fro": error,
            "gram_diagonal_mean": float(np.diag(gram).mean()),
        })

    axes[0, 0].set_title("Projected Gaussian distributions (2σ)")
    axes[0, 0].set_xlabel("shared component 1")
    axes[0, 0].set_ylabel("shared component 2")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].set_aspect("equal", adjustable="datalim")

    axes[0, 1].set_title("Learned low-rank eigenvalues")
    axes[0, 1].set_xlabel("sorted rank")
    axes[0, 1].set_ylabel("eigenvalue λ")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].set_title("Projected Stiefel frame directions")
    axes[1, 0].set_xlabel("shared component 1")
    axes[1, 0].set_ylabel("shared component 2")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].set_aspect("equal", adjustable="datalim")

    errors = [item["orthogonality_error_fro"] for item in diagnostics["items"]]
    axes[1, 1].bar(np.arange(count), errors, color=colors[:count])
    axes[1, 1].set_xticks(np.arange(count), labels, rotation=35, ha="right", fontsize=8)
    axes[1, 1].set_title("Orthonormal-frame error ‖VᵀV − I‖F")
    axes[1, 1].set_ylabel("Frobenius error")
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    output_path.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2))
    return diagnostics


def _load_model_and_tokenizer(args):
  

    if args.model_name == "bert-base-uncased" and args.tokenizer_path:
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    else:
        model_id = args.tokenizer_path or (
            "sentence-transformers/all-MiniLM-L6-v2"
            if args.model_name == "all-MiniLM-L6-v2" else args.model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    model = GaussianEmbeddingTuner(args).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--passage", action="append", required=True)
    parser.add_argument("--output", default="visualizations/finetune/query_passage.png")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=384)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--gaussian-eps", type=float, default=1e-6)
    parser.add_argument("--decomposition-type", default="stiefel_projected_lambdas")
    parser.add_argument("--stiefel-iterations", type=int, default=4)
    parser.add_argument("--min-variance", type=float, default=1e-5)
    parser.add_argument("--max-variance", type=float, default=10.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    args.cuda = args.device.startswith("cuda")

    model, tokenizer = _load_model_and_tokenizer(args)
    texts = [args.query, *args.passage]
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(args.device)
    with torch.no_grad():
        means, lambdas, frames, diagonals, _ = model.gaussian_projection(encoded)
    visualize_projection_batch(
        args.output,
        labels=["query", *[f"passage {i + 1}" for i in range(len(args.passage))]],
        means=means,
        lambdas=lambdas,
        frames=frames,
        diagonal_matrices=diagonals,
        title="Query and passage Gaussian geometry",
        dpi=args.dpi,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
