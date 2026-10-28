"""Tests for the regular-grid axisymmetric magnetic-field provider."""

from __future__ import annotations

import numpy as np
import pytest

from ecstartup.config import GeometryConfig
from ecstartup.physics.tabulated_field import (
    TabulatedFieldProvider,
    TabulatedMagneticField,
)
from ecstartup.physics import TabulatedMagneticField as PublicTabulatedMagneticField


def _linear_map():
    r = np.asarray([0.20, 0.40, 0.60, 0.80])
    z = np.asarray([-0.30, -0.10, 0.10, 0.30])
    R, Z = np.meshgrid(r, z, indexing="ij")
    # A linear map is reproduced exactly by RegularGridInterpolator.
    br = 0.20 + 2.0 * R - 0.5 * Z
    bz = -0.10 + 0.7 * R + 1.5 * Z
    bphi = 0.4 + 0.1 * R - 0.2 * Z
    return r, z, br, bz, bphi


def test_regular_grid_interpolation_and_static_protocol():
    r, z, br, bz, bphi = _linear_map()
    field = TabulatedMagneticField(
        r,
        z,
        br,
        bz,
        bphi,
        geometry=GeometryConfig(R0_m=0.4, minor_radius_m=0.15),
    )
    point = field.evaluate(0.35, 0.05, Ip_A=123.0, t_s=9.0)
    assert point.B_R_T == pytest.approx(0.20 + 2.0 * 0.35 - 0.5 * 0.05)
    assert point.B_Z_T == pytest.approx(-0.10 + 0.7 * 0.35 + 1.5 * 0.05)
    assert point.B_phi_T == pytest.approx(0.4 + 0.1 * 0.35 - 0.2 * 0.05)
    assert field.depends_on_ip is False
    assert field.depends_on_time is False
    assert field.vertical_field() == pytest.approx(-0.10 + 0.7 * 0.4)


def test_boundary_strategies_are_explicit():
    r, z, br, bz, bphi = _linear_map()
    with pytest.raises(ValueError, match="超出"):
        TabulatedMagneticField(r, z, br, bz, bphi).evaluate(0.1, 0.0, 0.0)

    clipped = TabulatedMagneticField(r, z, br, bz, bphi, boundary="nearest")
    assert clipped.evaluate(0.1, 0.0, 0.0).B_Z_T == pytest.approx(-0.10 + 0.7 * r[0] + 1.5 * 0.0)

    extrapolated = TabulatedMagneticField(r, z, br, bz, bphi, boundary="extrapolate")
    assert extrapolated.evaluate(0.1, 0.0, 0.0).B_R_T == pytest.approx(0.20 + 2.0 * 0.1, abs=1.0e-12)

    masked = TabulatedMagneticField(r, z, br, bz, bphi, boundary="nan")
    assert np.isnan(masked.evaluate(0.1, 0.0, 0.0).B_Z_T)


def test_supplied_psi_is_interpolated_and_alias_is_consistent():
    r, z, br, bz, bphi = _linear_map()
    R, Z = np.meshgrid(r, z, indexing="ij")
    psi = 0.5 * R**2 * (0.7) + 1.5 * R * Z
    field = TabulatedMagneticField(r, z, br, bz, bphi, psi_Wb_per_rad=psi)
    assert field.psi_source == "provided"
    # R**2 is intentionally not linear, so the interpolated value is checked
    # against a direct RegularGridInterpolator result rather than the
    # underlying analytic expression.
    from scipy.interpolate import RegularGridInterpolator

    expected = float(RegularGridInterpolator((r, z), psi)([[0.35, 0.05]])[0])
    assert field.flux_function(0.35, 0.05, 0.0) == pytest.approx(expected)
    values = field.flux_function(np.asarray([0.25, 0.55]), np.asarray([-0.2, 0.2]), 0.0)
    assert values.shape == (2,)
    assert np.isfinite(values).all()


def test_missing_psi_can_be_reconstructed_or_rejected():
    r = np.asarray([0.2, 0.4, 0.6])
    z = np.asarray([-0.1, 0.1])
    R, _ = np.meshgrid(r, z, indexing="ij")
    bz = 0.8 * np.ones_like(R)
    zeros = np.zeros_like(R)
    field = TabulatedMagneticField(r, z, zeros, bz, zeros)
    # Gauge is arbitrary; differences obey dpsi/dR = R Bz.
    delta = field.flux_function(0.6, 0.0, 0.0) - field.flux_function(0.2, 0.0, 0.0)
    assert delta == pytest.approx(0.5 * 0.8 * (0.6**2 - 0.2**2))
    assert field.diagnostics(0.0)["psi_source"] == "derived_from_B_Z"

    unavailable = TabulatedMagneticField(r, z, zeros, bz, zeros, derive_psi=False)
    with pytest.raises(ValueError, match="没有 psi"):
        unavailable.flux_function(0.4, 0.0, 0.0)


def test_validation_and_state_metadata_are_auditable():
    r, z, br, bz, bphi = _linear_map()
    with pytest.raises(ValueError):
        TabulatedMagneticField([0.2, 0.2], z, br[:2], bz[:2], bphi[:2])
    with pytest.raises(ValueError):
        TabulatedMagneticField(r, z, br[:, :2], bz[:, :2], bphi[:, :2])

    field = TabulatedMagneticField(
        r,
        z,
        br,
        bz,
        bphi,
        configuration="TPC-map",
        mirror_ratio=2.0,
        trapped_fraction=0.3,
        connection_length_factor=4.0,
        loss_multiplier=3.0,
    )
    state = field.configuration_state(0.0)
    assert state.configuration == "TPC-map"
    assert state.mirror_ratio == pytest.approx(2.0)
    assert state.trapped_fraction == pytest.approx(0.3)
    assert state.connection_length_factor == pytest.approx(4.0)
    assert state.loss_multiplier == pytest.approx(3.0)
    diagnostics = field.diagnostics(0.0)
    assert diagnostics["field_provider"] == "tabulated_regular_grid"
    assert diagnostics["grid_R_count"] == pytest.approx(len(r))
    assert diagnostics["grid_Z_count"] == pytest.approx(len(z))
    assert diagnostics["boundary_strategy"] == "raise"
    assert diagnostics["Bp_edge_provider_T"] > 0.0


def test_mapping_and_npz_loaders(tmp_path):
    r, z, br, bz, bphi = _linear_map()
    payload = {"R": r, "Z": z, "BR": br, "BZ": bz, "Bphi": bphi}
    field = TabulatedFieldProvider.from_mapping(payload, boundary="clip")
    assert isinstance(field, TabulatedMagneticField)
    assert field.evaluate(0.4, 0.0, 0.0).B_phi_T == pytest.approx(0.4 + 0.1 * 0.4)

    path = tmp_path / "field_map.npz"
    np.savez(path, **payload)
    loaded = TabulatedMagneticField.from_npz(path)
    assert loaded.evaluate(0.4, 0.0, 0.0).B_R_T == pytest.approx(0.20 + 2.0 * 0.4)


def test_public_export_and_xarray_named_dimension_transpose():
    xr = pytest.importorskip("xarray")
    r, z, br, bz, bphi = _linear_map()
    # Deliberately store variables as (Z, R), the common ordering in exported
    # equilibrium files.  The loader must use named dimensions rather than
    # assuming the raw array order.
    dataset = xr.Dataset(
        data_vars={
            "BR": (("Z", "R"), br.T),
            "BZ": (("Z", "R"), bz.T),
            "Bphi": (("Z", "R"), bphi.T),
        },
        coords={"R": r, "Z": z},
    )
    field = PublicTabulatedMagneticField.from_xarray(dataset)
    point = field.evaluate(0.35, 0.05, Ip_A=0.0)
    assert point.B_R_T == pytest.approx(0.20 + 2.0 * 0.35 - 0.5 * 0.05)
    assert point.B_Z_T == pytest.approx(-0.10 + 0.7 * 0.35 + 1.5 * 0.05)
