import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from finetune.offline_z_analysis import (
    analyze_geometry,
    load_geometry,
    orthogonal_polynomial_fit,
    quantile_profiles,
    sample_gaussians,
    synthetic_geometry,
)


def test_polynomial_reconstruction_error_decreases_with_order():
    geometry = synthetic_geometry(items=3, dim=32, rank=4)
    probabilities, profiles = quantile_profiles(geometry.z, points=65)
    for family in ("hermite", "chebyshev"):
        fit = orthogonal_polynomial_fit(profiles, probabilities, family, max_order=10)
        differences = np.diff(fit["relative_rmse"], axis=1)
        assert np.all(differences <= 1e-12)
        assert fit["coefficient_energy"].shape == (3, 11)


def test_low_rank_sampler_matches_target_moments():
    geometry = synthetic_geometry(items=2, dim=12, rank=3)
    clouds = sample_gaussians(geometry, 40_000, np.random.default_rng(11))
    expected_covariance = (
        np.diag(geometry.diagonals[0])
        + (geometry.frames[0] * geometry.lambdas[0]) @ geometry.frames[0].T
    )
    assert np.allclose(clouds[0].mean(axis=0), geometry.means[0], atol=0.025)
    assert np.allclose(np.cov(clouds[0], rowvar=False), expected_covariance, atol=0.025)


def test_offline_report_writes_figures_json_and_reusable_geometry(tmp_path):
    geometry = synthetic_geometry(items=4, dim=24, rank=4)
    report = analyze_geometry(
        geometry,
        tmp_path,
        samples=48,
        basis_order=7,
        density_bins=32,
        sliced_directions=8,
        bootstrap_repeats=4,
        bootstrap_samples=32,
        dpi=60,
    )

    for filename in (
        "z_basis.png",
        "gaussian_samples.png",
        "retrieval.png",
        "analysis.json",
        "geometry.npz",
    ):
        path = tmp_path / filename
        assert path.exists() and path.stat().st_size > 0

    saved_report = json.loads((tmp_path / "analysis.json").read_text())
    reloaded = load_geometry(tmp_path / "geometry.npz")
    assert saved_report["method"]["scope"] == "finetune only"
    assert len(saved_report["retrieval"]["records"]) == 3
    assert 0.0 <= report["retrieval"]["positive_top1_rate"] <= 1.0
    ellipses = saved_report["ellipses"]
    assert ellipses["coordinate_system"] == "independent local covariance plane per item"
    assert len(ellipses["items"]) == 4
    assert ellipses["standard_deviations"] == 2.0
    for item in ellipses["items"]:
        ellipse_path = tmp_path / item["file"]
        assert ellipse_path.exists() and ellipse_path.stat().st_size > 0
        assert np.asarray(item["axis_lengths"]).min() > 0
        assert np.asarray(item["eigen_residual_norms"]).max() < 1e-8
        assert "cannot be compared" in item["coordinate_note"]
    assert reloaded.labels == geometry.labels
    assert np.allclose(reloaded.z, geometry.z)
