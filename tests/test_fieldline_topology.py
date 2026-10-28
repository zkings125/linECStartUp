"""Tests for the optional field-line topology scan adapter."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from ecstartup.config import GeometryConfig
from ecstartup.physics.fieldline import FieldLineBounds, FieldLineTracer
from ecstartup.physics.fieldline_topology import (
    FieldLineTopologyDetector,
    FieldLineTopologyError,
    FieldLineTopologyScanner,
)
from ecstartup.physics.magnetic import CircularMagneticField


class ConstantVerticalField:
    configuration = "constant_vertical"

    def __init__(self, bounds: FieldLineBounds) -> None:
        self.bounds = bounds
        self.geometry = None

    def evaluate(self, R_m, Z_m, Ip_A, t_s=0.0):
        return SimpleNamespace(B_R_T=0.0, B_Z_T=1.0, B_phi_T=0.0)


class SingularTracer:
    """A tracer fixture with one deliberately singular seed."""

    def trace(self, R_m, Z_m, *, Ip_A=0.0, t_s=0.0, direction="both", start_phi_rad=0.0):
        if abs(R_m - 0.5) < 1.0e-12:
            raise ValueError("不能在理想细丝线圈本身的位置评估磁场（场发散）")
        return SimpleNamespace(
            status="closed",
            closed=True,
            length_m=2.0,
            mirror_ratio=3.0,
            min_B_T=1.0,
            max_B_T=3.0,
        )


def test_area_weighted_scan_reports_closed_and_effective_samples():
    geometry = GeometryConfig(B_vertical_T=0.0, R_bounds_m=(0.15, 0.85), Z_bounds_m=(-0.4, 0.4))
    field = CircularMagneticField(geometry)

    class ClosedTracer:
        def trace(self, R_m, Z_m, **kwargs):
            return SimpleNamespace(
                status="closed",
                closed=True,
                length_m=1.0,
                mirror_ratio=1.5,
                min_B_T=0.5,
                max_B_T=0.75,
            )

    tracer = ClosedTracer()
    scanner = FieldLineTopologyScanner(
        field,
        tracer,
        radial_fractions=(0.25, 0.75),
        poloidal_angles_rad=(0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0),
    )

    result = scanner.scan(100_000.0)

    assert result.total_sample_count == 8
    assert result.valid_sample_count == result.total_sample_count
    assert result.closed_sample_count == result.total_sample_count
    assert result.closed_fraction == pytest.approx(1.0)
    assert result.closed_area_fraction == pytest.approx(1.0)
    assert result.valid_fraction == pytest.approx(1.0)
    assert result.total_weight == pytest.approx(1.0)
    # Unequal annulus weights reduce the Kish effective sample count below
    # the raw eight seeds.
    assert result.effective_sample_count == pytest.approx(6.4)
    assert result.mean_connection_length_m == pytest.approx(1.0)
    assert result.mean_mirror_ratio == pytest.approx(1.5)
    assert len(result.to_dict()["samples"]) == 8


def test_invalid_zero_field_trajectories_are_excluded_from_fraction():
    bounds = FieldLineBounds((0.2, 0.8), (-0.1, 0.1))
    field = ConstantVerticalField(bounds)
    tracer = FieldLineTracer(field, bounds, step_m=0.03, max_length_m=1.0)
    # The custom provider is valid and reaches the wall for every seed.
    result = FieldLineTopologyScanner(
        field,
        tracer,
        config={"radial_fractions": (0.5,), "poloidal_angles_rad": (0.0, math.pi)},
    ).scan(0.0)
    assert result.valid_sample_count == 2
    assert result.closed_sample_count == 0
    assert result.closed_fraction == pytest.approx(0.0)
    assert result.status_counts["wall_hit"] == 2
    # ``direction=both`` reports the sum of the two one-sided connection
    # lengths, matching FieldLineTrace.length_m semantics.
    assert result.mean_connection_length_m == pytest.approx(0.2)


def test_singular_provider_error_is_recorded_and_raise_policy_is_available():
    geometry = GeometryConfig(R0_m=0.5, minor_radius_m=0.2, B_vertical_T=0.0)
    field = CircularMagneticField(geometry)
    scanner = FieldLineTopologyScanner(
        field,
        SingularTracer(),
        radial_fractions=(0.0, 0.5),
        poloidal_angles_rad=(0.0,),
    )
    result = scanner.scan(0.0)
    assert result.total_sample_count == 2
    assert result.invalid_sample_count == 1
    assert result.valid_sample_count == 1
    assert result.status_counts["singular_field"] == 1
    assert result.closed_fraction == pytest.approx(1.0)
    # The outer radial seed owns the annulus from rho=0.25 to 1, hence its
    # requested area weight is 15/16.
    assert result.valid_fraction == pytest.approx(0.9375)

    raising = FieldLineTopologyScanner(
        field,
        SingularTracer(),
        config={
            "radial_fractions": (0.0, 0.5),
            "poloidal_angles_rad": (0.0,),
            "invalid_policy": "raise",
        },
    )
    with pytest.raises(FieldLineTopologyError):
        raising.scan(0.0)


def test_detector_maps_scan_to_existing_topology_result_without_global_model():
    geometry = GeometryConfig(B_vertical_T=0.0)
    field = CircularMagneticField(geometry)
    class ClosedTracer:
        def trace(self, R_m, Z_m, **kwargs):
            return SimpleNamespace(
                status="closed",
                closed=True,
                length_m=1.0,
                mirror_ratio=1.0,
                min_B_T=0.5,
                max_B_T=0.5,
            )

    tracer = ClosedTracer()
    detector = FieldLineTopologyDetector(
        field,
        tracer,
        geometry=geometry,
        radial_seeds=2,
        poloidal_seeds=4,
        closed_fraction_threshold=0.5,
    )
    topology = detector.detect(100_000.0, 1.0e-4)
    assert topology.method.endswith("_fieldline_scan")
    assert topology.has_closed_surface
    assert topology.closed_fraction == pytest.approx(1.0)
    assert topology.connection_length_m > 0.0
    assert detector.last_scan is not None
