
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from finetune.model import GaussianEmbeddingTuner
from transformers import AutoTokenizer, BertTokenizer

REPORT_NOTE = (
    "The report is a representation diagnostic, not proof that multivector "
    "retrieval always improves relevance. A positive top-1 rate or rank "
    "improvement on labeled evaluation data is needed for that claim."
)


@dataclass
class Geometry:
    """The deterministic latent and Gaussian factors for query/candidates."""

    labels: list[str]
    z: np.ndarray
    means: np.ndarray
    lambdas: np.ndarray
    frames: np.ndarray
    diagonals: np.ndarray
    texts: list[str] | None = None
    positive_indices: list[int] | None = None

    def validate(self) -> "Geometry":
        arrays = [self.z, self.means, self.lambdas, self.frames, self.diagonals]
        count = len(self.labels)
        if count < 2:
            raise ValueError("Need at least a query and one candidate.")
        if any(len(value) != count for value in arrays):
            raise ValueError("labels, z, means, lambdas, frames, and diagonals must align")
        if self.z.ndim != 2 or self.means.ndim != 2:
            raise ValueError("z and means must have shape [items, dimensions]")
        if self.frames.ndim != 3 or self.lambdas.ndim != 2:
            raise ValueError("frames/lambdas must have shapes [items, dim, rank]/[items, rank]")
        if self.frames.shape[1:] != (self.means.shape[1], self.lambdas.shape[1]):
            raise ValueError("frame dimensions do not match means and lambdas")
        if self.diagonals.ndim == 3:
            self.diagonals = np.diagonal(self.diagonals, axis1=-2, axis2=-1)
        if self.diagonals.shape != self.means.shape:
            raise ValueError("diagonals must have shape [items, dimensions] (or diagonal matrices)")
        if np.any(self.lambdas < 0) or np.any(self.diagonals < 0):
            raise ValueError("Gaussian variances must be non-negative")
        if self.texts is not None and len(self.texts) != count:
            raise ValueError("texts must align with labels")
        self.positive_indices = list(self.positive_indices or [1])
        if any(index <= 0 or index >= count for index in self.positive_indices):
            raise ValueError("positive_indices refer to candidate item indices, so they must be >= 1")
        return self


def _array(value: Any) -> np.ndarray:
    """Convert NumPy or torch-like values to float64 NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def save_geometry(path: str | Path, geometry: Geometry) -> None:
    """Save model outputs so all later analysis is transformer-free."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.asarray(geometry.labels, dtype=str),
        texts=np.asarray(geometry.texts or geometry.labels, dtype=str),
        positive_indices=np.asarray(geometry.positive_indices, dtype=np.int64),
        z=geometry.z,
        means=geometry.means,
        lambdas=geometry.lambdas,
        frames=geometry.frames,
        diagonals=geometry.diagonals,
    )


def load_geometry(path: str | Path) -> Geometry:
    """Load a ``geometry.npz`` artifact produced by this script."""
    with np.load(path, allow_pickle=False) as data:
        required = {"labels", "z", "means", "lambdas", "frames", "diagonals"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {', '.join(sorted(missing))}")
        geometry = Geometry(
            labels=data["labels"].astype(str).tolist(),
            texts=data["texts"].astype(str).tolist() if "texts" in data else None,
            positive_indices=(
                data["positive_indices"].astype(int).tolist()
                if "positive_indices" in data else [1]
            ),
            z=_array(data["z"]),
            means=_array(data["means"]),
            lambdas=_array(data["lambdas"]),
            frames=_array(data["frames"]),
            diagonals=_array(data["diagonals"]),
        )
    return geometry.validate()


def synthetic_geometry(items: int = 5, dim: int = 64, rank: int = 8, seed: int = 7) -> Geometry:
    """Create deterministic geometry for a dependency-free demonstration."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(items, dim))
    z[0] += 0.35 * np.sin(np.linspace(0, 3 * np.pi, dim))
    means = rng.normal(scale=0.55, size=(items, dim))
    means[1] = means[0] + rng.normal(scale=0.08, size=dim)
    raw_frames = rng.normal(size=(items, dim, rank))
    frames = np.stack([np.linalg.qr(frame, mode="reduced")[0] for frame in raw_frames])
    lambdas = rng.lognormal(mean=-1.0, sigma=0.55, size=(items, rank))
    diagonals = rng.lognormal(mean=-1.8, sigma=0.35, size=(items, dim))
    # Make the relevant passage similar in both center and uncertainty.
    frames[1] = frames[0] + 0.03 * rng.normal(size=(dim, rank))
    frames[1] = np.linalg.qr(frames[1], mode="reduced")[0]
    lambdas[1] = lambdas[0] * rng.lognormal(0.0, 0.05, size=rank)
    diagonals[1] = diagonals[0] * rng.lognormal(0.0, 0.05, size=dim)
    labels = ["query", "positive", *[f"negative {i}" for i in range(1, items - 1)]]
    return Geometry(
        labels=labels,
        texts=labels,
        positive_indices=[1],
        z=z,
        means=means,
        lambdas=lambdas,
        frames=frames,
        diagonals=diagonals,
    ).validate()


def quantile_profiles(values: np.ndarray, points: int = 129) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized empirical quantile functions for a batch of vectors."""
    values = _array(values)
    if values.ndim != 2:
        raise ValueError("values must have shape [items, dimensions]")
    probabilities = np.linspace(0.005, 0.995, points)
    profiles = np.empty((len(values), points), dtype=np.float64)
    for index, row in enumerate(values):
        scale = row.std()
        standardized = (row - row.mean()) / (scale if scale > 1e-12 else 1.0)
        profiles[index] = np.quantile(standardized, probabilities)
    return probabilities, profiles


def orthogonal_polynomial_fit(
    profiles: np.ndarray,
    probabilities: np.ndarray,
    family: str,
    max_order: int,
) -> dict[str, np.ndarray]:
    """Fit profiles in a QR-orthonormalized Hermite or Chebyshev basis.

    QR makes coefficient energy comparable by polynomial order.  Hermite uses
    the standard-normal quantile coordinate; Chebyshev uses ``[-1, 1]``.
    """
    profiles = _array(profiles)
    probabilities = _array(probabilities)
    order = min(max_order, len(probabilities) - 1)
    if family == "hermite":
        normal = NormalDist()
        coordinate = np.asarray([normal.inv_cdf(float(p)) for p in probabilities])
        design = np.polynomial.hermite_e.hermevander(coordinate, order)
    elif family == "chebyshev":
        coordinate = 2.0 * probabilities - 1.0
        design = np.polynomial.chebyshev.chebvander(coordinate, order)
    else:
        raise ValueError("family must be 'hermite' or 'chebyshev'")

    q_basis, _ = np.linalg.qr(design, mode="reduced")
    coefficients = profiles @ q_basis
    total_energy = np.sum(profiles**2, axis=1, keepdims=True).clip(min=1e-12)
    coefficient_energy = coefficients**2 / total_energy
    relative_rmse = []
    for terms in range(1, order + 2):
        reconstruction = coefficients[:, :terms] @ q_basis[:, :terms].T
        numerator = np.mean((profiles - reconstruction) ** 2, axis=1)
        denominator = np.mean(profiles**2, axis=1).clip(min=1e-12)
        relative_rmse.append(np.sqrt(numerator / denominator))
    return {
        "coordinate": coordinate,
        "basis": q_basis,
        "coefficients": coefficients,
        "coefficient_energy": coefficient_energy,
        "relative_rmse": np.stack(relative_rmse, axis=1),
        "reconstruction": coefficients @ q_basis.T,
    }


def sample_gaussians(geometry: Geometry, samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from D + V diag(lambda) V^T without forming dense covariance."""
    diagonal_noise = rng.normal(size=(len(geometry.labels), samples, geometry.means.shape[1]))
    diagonal_part = diagonal_noise * np.sqrt(geometry.diagonals)[:, None, :]
    rank_noise = rng.normal(size=(len(geometry.labels), samples, geometry.lambdas.shape[1]))
    scaled_rank_noise = rank_noise * np.sqrt(geometry.lambdas)[:, None, :]
    low_rank_part = np.einsum("nsr,ndr->nsd", scaled_rank_noise, geometry.frames)
    return geometry.means[:, None, :] + diagonal_part + low_rank_part




def _covariance_matmul(
    frame: np.ndarray,
    lambdas: np.ndarray,
    diagonal: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    """Multiply (D + V diag(lambda) V^T) by one or more vectors."""
    return diagonal[:, None] * vectors + frame @ (
        lambdas[:, None] * (frame.T @ vectors)
    )


def learned_local_ellipse_planes(
    geometry: Geometry,
    iterations: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find each Gaussian's own top-two covariance directions.

    No coordinate system is shared between items. For large embeddings a
    deterministic subspace iteration avoids materializing dense covariance.
    """
    dimension = geometry.means.shape[1]
    if dimension < 2:
        raise ValueError("Ellipse visualization requires output_dim >= 2")
    planes = []
    variances = []
    residuals = []
    rng = np.random.default_rng(0)

    for frame, lambdas, diagonal in zip(
        geometry.frames, geometry.lambdas, geometry.diagonals
    ):
        if dimension <= 256:
            covariance = np.diag(diagonal) + (frame * lambdas[None, :]) @ frame.T
            values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
            plane = vectors[:, -2:][:, ::-1]
            values = values[-2:][::-1]
        else:
            subspace_size = min(dimension, max(8, frame.shape[1] + 2))
            initial = rng.normal(size=(dimension, subspace_size))
            retained = min(frame.shape[1], subspace_size - 2)
            initial[:, :retained] = frame[:, :retained]
            for offset, coordinate in enumerate(np.argsort(diagonal)[-2:]):
                initial[:, retained + offset] = 0.0
                initial[coordinate, retained + offset] = 1.0
            subspace, _ = np.linalg.qr(initial, mode="reduced")
            for _ in range(iterations):
                subspace, _ = np.linalg.qr(
                    _covariance_matmul(frame, lambdas, diagonal, subspace),
                    mode="reduced",
                )
            reduced = subspace.T @ _covariance_matmul(
                frame, lambdas, diagonal, subspace
            )
            values, vectors = np.linalg.eigh((reduced + reduced.T) / 2.0)
            plane = (subspace @ vectors[:, -2:])[:, ::-1]
            values = values[-2:][::-1]

        covariance_plane = _covariance_matmul(frame, lambdas, diagonal, plane)
        residual = covariance_plane - plane * values[None, :]
        planes.append(plane)
        variances.append(np.clip(values, 1e-12, None))
        residuals.append(np.linalg.norm(residual, axis=0))

    return np.stack(planes), np.stack(variances), np.stack(residuals)




def density_profiles(projected_1d: np.ndarray, bins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Estimate comparable densities on a common robust range."""
    projected_1d = _array(projected_1d)
    low, high = np.quantile(projected_1d, [0.005, 0.995])
    if high - low < 1e-12:
        low, high = low - 1.0, high + 1.0
    edges = np.linspace(low, high, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    densities = np.stack([
        np.histogram(row, bins=edges, density=True)[0] for row in projected_1d
    ])
    return centers, densities


def fourier_power(profiles: np.ndarray) -> np.ndarray:
    """Return normalized non-DC real-Fourier power for each profile."""
    centered = profiles - profiles.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    power[:, 0] = 0.0
    return power / power.sum(axis=1, keepdims=True).clip(min=1e-12)


def haar_scale_energy(profiles: np.ndarray) -> np.ndarray:
    """Return normalized Haar detail energy from fine to coarse scales."""
    length = 2 ** int(np.floor(np.log2(profiles.shape[1])))
    approximation = profiles[:, :length].copy()
    energies = []
    while approximation.shape[1] >= 2:
        even = approximation[:, 0::2]
        odd = approximation[:, 1::2]
        detail = (even - odd) / np.sqrt(2.0)
        approximation = (even + odd) / np.sqrt(2.0)
        energies.append(np.sum(detail**2, axis=1))
    result = np.stack(energies, axis=1)
    return result / result.sum(axis=1, keepdims=True).clip(min=1e-12)


def model_bures_components(geometry: Geometry) -> list[dict[str, float | int | str | bool]]:
    """Reproduce the low-rank/diagonal Bures surrogate used in model.py.

    The implementation in ``GaussianEmbeddingTuner.exact_bures_wasserstein``
    decomposes diagonal and low-rank cross terms.  This function names it a
    model score rather than claiming it is the dense-covariance closed form.
    """
    query = 0
    records: list[dict[str, float | int | str | bool]] = []
    for candidate in range(1, len(geometry.labels)):
        mean_energy = float(np.sum((geometry.means[query] - geometry.means[candidate]) ** 2))
        trace_q = float(geometry.lambdas[query].sum() + geometry.diagonals[query].sum())
        trace_p = float(geometry.lambdas[candidate].sum() + geometry.diagonals[candidate].sum())
        diagonal_cross = float(np.sqrt(np.clip(
            geometry.diagonals[query] * geometry.diagonals[candidate],
            0.0,
            None,
        )).sum())
        cross = geometry.frames[query].T @ geometry.frames[candidate]
        scaled_cross = (
            np.sqrt(geometry.lambdas[query])[:, None]
            * cross
            * np.sqrt(geometry.lambdas[candidate])[None, :]
        )
        low_rank_cross = float(np.linalg.svd(scaled_cross, compute_uv=False).sum())
        covariance_energy = trace_q + trace_p - 2.0 * (diagonal_cross + low_rank_cross)
        records.append({
            "item_index": candidate,
            "label": geometry.labels[candidate],
            "relevant": candidate in geometry.positive_indices,
            "mean_squared_distance": mean_energy,
            "covariance_energy": covariance_energy,
            "model_bures_squared": max(0.0, mean_energy + covariance_energy),
        })
    return records


def sliced_wasserstein_squared(
    query_samples: np.ndarray,
    candidate_samples: np.ndarray,
    directions: np.ndarray,
) -> float:
    """Monte Carlo sliced-W2 score between equally sized multivector sets."""
    q_projection = np.sort(query_samples @ directions.T, axis=0)
    p_projection = np.sort(candidate_samples @ directions.T, axis=0)
    return float(np.mean((q_projection - p_projection) ** 2))


def bootstrap_multivector_scores(
    geometry: Geometry,
    samples_per_cloud: int,
    directions: int,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeatedly sample each Gaussian and score candidates with sliced W2."""
    random_directions = rng.normal(size=(directions, geometry.means.shape[1]))
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True).clip(min=1e-12)
    scores = np.empty((repeats, len(geometry.labels) - 1), dtype=np.float64)
    top1 = np.empty(repeats, dtype=np.int64)
    for repeat in range(repeats):
        clouds = sample_gaussians(geometry, samples_per_cloud, rng)
        for offset, candidate in enumerate(range(1, len(geometry.labels))):
            scores[repeat, offset] = sliced_wasserstein_squared(
                clouds[0], clouds[candidate], random_directions
            )
        top1[repeat] = int(np.argmin(scores[repeat]) + 1)
    return scores, top1, random_directions


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _plot_z_basis(
    output: Path,
    geometry: Geometry,
    probabilities: np.ndarray,
    profiles: np.ndarray,
    hermite: dict[str, np.ndarray],
    chebyshev: dict[str, np.ndarray],
    dpi: int,
) -> None:
    plt = _matplotlib()
    colors = plt.cm.tab10(np.linspace(0, 1, len(geometry.labels)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for index, label in enumerate(geometry.labels):
        axes[0, 0].plot(probabilities, profiles[index], label=label, color=colors[index])
        axes[0, 1].plot(
            np.arange(1, hermite["relative_rmse"].shape[1] + 1),
            hermite["relative_rmse"][index], color=colors[index], label=label,
        )
        axes[1, 0].plot(
            np.arange(1, chebyshev["relative_rmse"].shape[1] + 1),
            chebyshev["relative_rmse"][index], color=colors[index], label=label,
        )
    axes[0, 0].set(title="Standardized z empirical quantile functions", xlabel="percentile", ylabel="z-score")
    axes[0, 1].set(title="Hermite approximation", xlabel="number of basis terms", ylabel="relative RMSE")
    axes[1, 0].set(title="Chebyshev approximation", xlabel="number of basis terms", ylabel="relative RMSE")
    max_terms = min(12, hermite["coefficient_energy"].shape[1])
    x = np.arange(max_terms)
    width = 0.8 / len(geometry.labels)
    for index, label in enumerate(geometry.labels):
        axes[1, 1].bar(
            x + (index - (len(geometry.labels) - 1) / 2) * width,
            hermite["coefficient_energy"][index, :max_terms],
            width=width, color=colors[index], label=label,
        )
    axes[1, 1].set(title="Hermite coefficient energy in z", xlabel="polynomial order", ylabel="fraction of profile energy")
    axes[1, 1].set_xticks(x)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Latent z functional-basis analysis")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_gaussian_samples(
    output: Path,
    geometry: Geometry,
    standardized_local_pc1: np.ndarray,
    density_x: np.ndarray,
    densities: np.ndarray,
    power: np.ndarray,
    wavelet_energy: np.ndarray,
    dpi: int,
) -> None:
    plt = _matplotlib()
    colors = plt.cm.tab10(np.linspace(0, 1, len(geometry.labels)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    percentiles = np.linspace(0.01, 0.99, 99)
    for index, label in enumerate(geometry.labels):
        axes[0, 0].plot(
            percentiles, np.quantile(standardized_local_pc1[index], percentiles),
            color=colors[index], label=label,
        )
        axes[0, 1].plot(density_x, densities[index], color=colors[index], label=label)
        visible_frequencies = min(32, power.shape[1] - 1)
        axes[1, 0].plot(
            np.arange(1, visible_frequencies + 1), power[index, 1:visible_frequencies + 1],
            color=colors[index], label=label,
        )
        axes[1, 1].plot(
            np.arange(1, wavelet_energy.shape[1] + 1), wavelet_energy[index],
            marker="o", markersize=3, color=colors[index], label=label,
        )
    axes[0, 0].set(title="Sample quantiles on each local covariance direction 1",
                   xlabel="percentile", ylabel="standardized local displacement")
    axes[0, 1].set(title="Density on each item's own covariance direction 1",
                   xlabel="standard deviations from its own μ", ylabel="density")
    axes[1, 0].set(title="Fourier power of PC-1 density", xlabel="frequency bin", ylabel="normalized power")
    axes[1, 1].set(title="Haar-wavelet density energy", xlabel="scale (fine to coarse)", ylabel="normalized detail energy")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Sampled Gaussian functional and multivector view")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _safe_label_filename(label: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in label)
    return cleaned.strip("_").lower() or "item"


def _plot_learned_ellipses(
    output_dir: Path,
    geometry: Geometry,
    clouds: np.ndarray,
    planes: np.ndarray,
    variances: np.ndarray,
    residuals: np.ndarray,
    standard_deviations: float,
    dpi: int,
) -> list[dict[str, Any]]:
    """Write one local-principal-plane ellipse plot for each Gaussian."""
    plt = _matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    unit_circle = np.stack([np.cos(angles), np.sin(angles)])
    diagnostics = []

    for index, label in enumerate(geometry.labels):
        projected_samples = (clouds[index] - geometry.means[index]) @ planes[index]
        outer_lengths = standard_deviations * np.sqrt(variances[index])
        filename = f"{index:03d}_{_safe_label_filename(label)}.png"
        figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
        axis.scatter(
            projected_samples[:, 0], projected_samples[:, 1],
            color="#4c78a8", s=10, alpha=0.18, label="Gaussian samples",
        )
        levels = [1.0]
        if not np.isclose(standard_deviations, 1.0):
            levels.append(standard_deviations)
        for level in levels:
            contour = np.sqrt(variances[index])[:, None] * unit_circle * level
            axis.plot(
                contour[0], contour[1], linewidth=2.2,
                linestyle="--" if level == 1.0 and len(levels) > 1 else "-",
                label=f"{level:g}σ learned-covariance ellipse",
            )
        axis.scatter(0.0, 0.0, color="black", marker="x", s=75, label="learned μ")
        limit = max(
            float(outer_lengths.max()),
            float(np.quantile(np.abs(projected_samples), 0.995)),
            1e-6,
        ) * 1.15
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set(
            title=(
                f"{label}: learned Gaussian ellipse\n"
                f"local plane centered at μ; ‖μ‖={np.linalg.norm(geometry.means[index]):.3g}"
            ),
            xlabel="local covariance direction 1: displacement from μ",
            ylabel="local covariance direction 2: displacement from μ",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        figure.savefig(output_dir / filename, dpi=dpi)
        plt.close(figure)
        diagnostics.append({
            "label": label,
            "file": f"ellipses/{filename}",
            "mean_norm": float(np.linalg.norm(geometry.means[index])),
            "principal_variances": variances[index].tolist(),
            "axis_lengths": outer_lengths.tolist(),
            "eigen_residual_norms": residuals[index].tolist(),
            "basis_orthogonality_error": float(
                np.linalg.norm(planes[index].T @ planes[index] - np.eye(2))
            ),
            "coordinate_note": (
                "Axes are this Gaussian's own top covariance directions and cannot be "
                "compared as coordinates with axes in another item's plot."
            ),
        })
    return diagnostics


def _plot_retrieval(
    output: Path,
    records: list[dict[str, Any]],
    bootstrap_scores: np.ndarray,
    dpi: int,
) -> None:
    plt = _matplotlib()
    labels = [str(record["label"]) for record in records]
    mean_scores = np.asarray([record["mean_squared_distance"] for record in records])
    covariance = np.asarray([record["covariance_energy"] for record in records])
    model_scores = np.asarray([record["model_bures_squared"] for record in records])
    sample_mean = bootstrap_scores.mean(axis=0)
    sample_std = bootstrap_scores.std(axis=0)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    axes[0].bar(x, mean_scores, label="mean distance", color="#4c78a8")
    axes[0].bar(x, covariance, bottom=mean_scores, label="covariance contribution", color="#f58518")
    axes[0].scatter(x, model_scores, marker="_", s=220, color="black", label="clamped model score")
    axes[0].set(title="Model retrieval score decomposition", ylabel="squared distance")
    axes[1].bar(x, sample_mean, yerr=sample_std, capsize=4, color="#54a24b")
    axes[1].set(title="Sampled multivector sliced-W2", ylabel="mean score ± bootstrap SD")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean-only versus uncertainty-aware retrieval (lower is better)")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install the visualization extra: pip install -e '.[viz]'") from exc
    return plt


def analyze_geometry(
    geometry: Geometry,
    output_dir: str | Path,
    *,
    samples: int = 256,
    basis_order: int = 12,
    density_bins: int = 128,
    sliced_directions: int = 64,
    bootstrap_repeats: int = 30,
    bootstrap_samples: int = 128,
    ellipse_standard_deviations: float = 2.0,
    seed: int = 20,
    dpi: int = 150,
) -> dict[str, Any]:
    """Run the complete offline report and return JSON-serializable metrics."""
    geometry = geometry.validate()
    if min(samples, density_bins, sliced_directions, bootstrap_repeats, bootstrap_samples) < 2:
        raise ValueError("sampling, bin, direction, and repeat counts must all be >= 2")
    if ellipse_standard_deviations <= 0:
        raise ValueError("ellipse_standard_deviations must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    probabilities, z_profiles = quantile_profiles(geometry.z)
    hermite = orthogonal_polynomial_fit(z_profiles, probabilities, "hermite", basis_order)
    chebyshev = orthogonal_polynomial_fit(z_profiles, probabilities, "chebyshev", basis_order)

    clouds = sample_gaussians(geometry, samples, rng)
    ellipse_planes, ellipse_variances, ellipse_residuals = (
        learned_local_ellipse_planes(geometry)
    )
    local_projected = np.einsum(
        "nsd,ndk->nsk", clouds - geometry.means[:, None, :], ellipse_planes
    )
    standardized_local_pc1 = local_projected[:, :, 0] / np.sqrt(ellipse_variances[:, 0])[:, None]
    density_x, densities = density_profiles(standardized_local_pc1, density_bins)
    power = fourier_power(densities)
    wavelet_energy = haar_scale_energy(densities)
    density_probabilities = np.linspace(0.005, 0.995, densities.shape[1])
    density_hermite = orthogonal_polynomial_fit(
        densities, density_probabilities, "hermite", basis_order
    )
    density_chebyshev = orthogonal_polynomial_fit(
        densities, density_probabilities, "chebyshev", basis_order
    )

    records = model_bures_components(geometry)
    bootstrap_scores, bootstrap_top1, _ = bootstrap_multivector_scores(
        geometry, bootstrap_samples, sliced_directions, bootstrap_repeats, rng
    )
    mean_ranks = _rank(np.asarray([record["mean_squared_distance"] for record in records]))
    model_ranks = _rank(np.asarray([record["model_bures_squared"] for record in records]))
    sample_ranks = _rank(bootstrap_scores.mean(axis=0))
    for offset, record in enumerate(records):
        item_index = int(record["item_index"])
        record.update({
            "mean_only_rank": int(mean_ranks[offset]),
            "model_bures_rank": int(model_ranks[offset]),
            "sampled_sliced_w2_mean": float(bootstrap_scores[:, offset].mean()),
            "sampled_sliced_w2_std": float(bootstrap_scores[:, offset].std()),
            "sampled_sliced_w2_rank": int(sample_ranks[offset]),
            "sampled_top1_rate": float(np.mean(bootstrap_top1 == item_index)),
        })

    _plot_z_basis(
        output_dir / "z_basis.png", geometry, probabilities, z_profiles,
        hermite, chebyshev, dpi,
    )
    _plot_gaussian_samples(
        output_dir / "gaussian_samples.png", geometry, standardized_local_pc1, density_x,
        densities, power, wavelet_energy, dpi,
    )
    ellipse_diagnostics = _plot_learned_ellipses(
        output_dir / "ellipses", geometry, clouds,
        ellipse_planes, ellipse_variances, ellipse_residuals,
        ellipse_standard_deviations, dpi,
    )
    _plot_retrieval(output_dir / "retrieval.png", records, bootstrap_scores, dpi)
    save_geometry(output_dir / "geometry.npz", geometry)

    final_term = min(basis_order + 1, hermite["relative_rmse"].shape[1]) - 1
    report: dict[str, Any] = {
        "method": {
            "scope": "finetune only",
            "z_analysis": (
                "Each z vector is converted to a sorted standardized empirical quantile "
                "function, so results do not depend on arbitrary embedding-coordinate order."
            ),
            "hermite": "Probabilists' Hermite polynomials on standard-normal quantile coordinates.",
            "chebyshev": "Chebyshev polynomials on the bounded percentile coordinate [-1, 1].",
            "spectral_analysis": (
                "Fourier and Haar-wavelet energy are computed on sampled Gaussian density "
                "profiles along each item's own first covariance direction, standardized "
                "by that item's learned variance. These are not shared coordinates."
            ),
            "ellipses": (
                "Each learned Gaussian is drawn separately in its own top-two covariance "
                "directions, centered at its own mean. No coordinate axes are shared "
                "between item plots."
            ),
            "retrieval": (
                "A learned Gaussian is a compact generator of multiple retrieval vectors. "
                "The model score uses center and covariance; sampled sliced-W2 independently "
                "checks the same distributional idea with explicit multivectors."
            ),
            "caution": REPORT_NOTE,
        },
        "configuration": {
            "seed": seed,
            "samples": samples,
            "basis_order": basis_order,
            "density_bins": density_bins,
            "sliced_directions": sliced_directions,
            "bootstrap_repeats": bootstrap_repeats,
            "bootstrap_samples": bootstrap_samples,
            "ellipse_standard_deviations": ellipse_standard_deviations,
        },
        "labels": geometry.labels,
        "positive_indices": geometry.positive_indices,
        "z": [],
        "sample_density": [],
        "ellipses": {
            "standard_deviations": ellipse_standard_deviations,
            "coordinate_system": "independent local covariance plane per item",
            "items": ellipse_diagnostics,
        },
        "retrieval": {
            "records": records,
            "positive_top1_rate": float(np.mean(np.isin(bootstrap_top1, geometry.positive_indices))),
        },
        "artifacts": {
            "z_basis": "z_basis.png",
            "gaussian_samples": "gaussian_samples.png",
            "retrieval": "retrieval.png",
            "ellipses": [item["file"] for item in ellipse_diagnostics],
            "geometry": "geometry.npz",
        },
    }
    for index, label in enumerate(geometry.labels):
        report["z"].append({
            "label": label,
            "mean": float(geometry.z[index].mean()),
            "std": float(geometry.z[index].std()),
            "hermite_relative_rmse": float(hermite["relative_rmse"][index, final_term]),
            "chebyshev_relative_rmse": float(chebyshev["relative_rmse"][index, final_term]),
            "hermite_coefficient_energy": hermite["coefficient_energy"][index].tolist(),
            "chebyshev_coefficient_energy": chebyshev["coefficient_energy"][index].tolist(),
        })
        report["sample_density"].append({
            "label": label,
            "hermite_relative_rmse": float(density_hermite["relative_rmse"][index, final_term]),
            "chebyshev_relative_rmse": float(density_chebyshev["relative_rmse"][index, final_term]),
            "fourier_power": power[index].tolist(),
            "haar_scale_energy_fine_to_coarse": wavelet_energy[index].tolist(),
        })
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2))
    return report


def _checkpoint_geometry(args: argparse.Namespace) -> Geometry:
    """Load a finetune checkpoint and extract z plus Gaussian factors."""
    

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

    texts = [args.query, args.positive, *(args.negative or [])]
    labels = ["query", "positive", *[f"negative {i + 1}" for i in range(len(args.negative or []))]]
    max_length = {
        "bert-base-uncased": 512,
        "all-MiniLM-L6-v2": 256,
        "bge-base-en-v1.5": 512,
    }.get(args.model_name, 8192)
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(args.device)
    with torch.no_grad():
        encoder_embedding = model.get_cls(encoded)
        z = model.z_transform(encoder_embedding)
        means = model.center_prediction(z)
        frames, lambdas, diagonal_matrices, _ = model.decomposed_covariance(z)
    return Geometry(
        labels=labels,
        texts=texts,
        positive_indices=[1],
        z=_array(z),
        means=_array(means),
        lambdas=_array(lambdas),
        frames=_array(frames),
        diagonals=_array(diagonal_matrices),
    ).validate()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Analyze deterministic synthetic geometry.")
    source.add_argument("--input-npz", help="Analyze an existing geometry.npz without loading a model.",required=False)
    source.add_argument("--checkpoint", help="Local finetune checkpoint to analyze.")
    parser.add_argument("--query")
    parser.add_argument("--positive")
    parser.add_argument("--negative", action="append", default=[])
    parser.add_argument("--output-dir", default="visualizations/finetune/offline_z")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--basis-order", type=int, default=12)
    parser.add_argument("--density-bins", type=int, default=128)
    parser.add_argument("--sliced-directions", type=int, default=64)
    parser.add_argument("--bootstrap-repeats", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)

    parser.add_argument("--ellipse-std", type=float, default=2.0,
                        help="Outer ellipse radius in local standard deviations.")
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
    if args.checkpoint and (not args.query or not args.positive):
        raise SystemExit("--checkpoint requires --query and --positive")
    if args.demo:
        geometry = synthetic_geometry(seed=args.seed)
    elif args.input_npz:
        geometry = load_geometry(args.input_npz)
    else:
        geometry = _checkpoint_geometry(args)
    report = analyze_geometry(
        geometry,
        args.output_dir,
        samples=args.samples,
        basis_order=args.basis_order,
        density_bins=args.density_bins,
        sliced_directions=args.sliced_directions,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        dpi=args.dpi,
        ellipse_standard_deviations=args.ellipse_std,
    )
    print(f"Wrote offline z/Gaussian analysis to {Path(args.output_dir).resolve()}")
    print(f"Relevant-candidate sampled top-1 rate: {report['retrieval']['positive_top1_rate']:.3f}")


if __name__ == "__main__":
    main()
