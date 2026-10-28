from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from ecstartup.physics.mat_equilibrium import read_mat_equilibrium


def _write_fixture(path: Path) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    R = 1.0 + 0.2 * np.cos(theta)
    Z = 0.3 * np.sin(theta)
    grid = np.linspace(0.8, 1.2, 9)
    Rg, Zg = np.meshgrid(grid, np.linspace(-0.3, 0.3, 9), indexing="ij")
    savemat(
        path,
        {
            "Bt": 1.7,
            "Ip": 2.0e5,
            "Raxis": 1.0,
            "Zaxis": 0.0,
            "R_LCFS": 1.0,
            "Z_LCFS": 0.0,
            "psi_axis": 0.0,
            "psi_LCFS": 1.0,
            "Ra": R,
            "Za": Z,
            "gridr": Rg,
            "gridz": Zg,
            "psi_total": (Rg - 1.0) ** 2 + Zg**2,
        },
    )


def test_mat_equilibrium_extracts_shape_only_reference(tmp_path: Path) -> None:
    path = tmp_path / "equilibrium.mat"
    _write_fixture(path)
    reference = read_mat_equilibrium(path)
    assert reference.B_t_T == pytest.approx(1.7)
    assert reference.minor_radius_m == pytest.approx(np.sqrt(0.2 * 0.3), rel=1.0e-3)
    assert reference.elongation > 1.0
    assert reference.summary()["reference_kind"] == "closed_surface_equilibrium_shape_only"
    assert "no pre-CFS magnetic reconstruction" in reference.summary()["startup_use"]


def test_user_jtext_mat_is_read_when_available() -> None:
    path = Path(r"D:\19673\Downloads\HuakeExperiment\EQ_data_1109520_340beta0.3.mat")
    if not path.exists():
        pytest.skip("user-provided J-TEXT MAT file is not mounted")
    reference = read_mat_equilibrium(path)
    assert 1.0 < reference.B_t_T < 3.0
    assert reference.Ip_A > 0.0
    assert 3.0 < reference.aspect_ratio < 7.0
