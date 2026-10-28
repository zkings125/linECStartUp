"""轴对称场线追踪器的解析回归测试。"""

from __future__ import annotations

import math

import pytest

from ecstartup.config import GeometryConfig
from ecstartup.physics.fieldline import FieldLineBounds, FieldLineTracer, trace_field_line
from ecstartup.physics.magnetic import CircularMagneticField, make_magnetic_field


class ConstantField:
    """Minimal provider used to isolate integration and wall logic."""

    configuration = "constant_test"

    def __init__(self, B_R_T: float = 0.0, B_Z_T: float = 0.0, B_phi_T: float = 0.0) -> None:
        self.B_R_T = B_R_T
        self.B_Z_T = B_Z_T
        self.B_phi_T = B_phi_T

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0):
        # Importing FieldPoint here would make this fixture depend on a
        # particular provider implementation; the tracer intentionally only
        # requires these three attributes.
        return type(
            "Point",
            (),
            {"B_R_T": self.B_R_T, "B_Z_T": self.B_Z_T, "B_phi_T": self.B_phi_T},
        )()


def test_circular_provider_closes_in_poloidal_projection():
    geometry = GeometryConfig(
        B_vertical_T=0.0,
        R_bounds_m=(0.15, 0.85),
        Z_bounds_m=(-0.40, 0.40),
    )
    field = CircularMagneticField(geometry)
    tracer = FieldLineTracer(
        field,
        step_m=0.005,
        max_length_m=3.0,
        closure_tolerance_m=0.003,
    )
    trace = tracer.trace(
        start=(geometry.R0_m + 0.10, 0.0),
        Ip_A=100_000.0,
        direction="both",
    )

    assert trace.status == "closed"
    assert trace.closed
    assert not trace.hit_boundary
    assert len(trace.branches) == 2
    assert trace.forward is not None and trace.backward is not None
    assert trace.forward.closed and trace.backward.closed
    assert trace.forward.length_m == pytest.approx(trace.backward.length_m, rel=2.0e-3)
    assert trace.length_m == pytest.approx(
        trace.forward.length_m + trace.backward.length_m,
    )
    assert trace.min_B_T > 0.0
    assert trace.max_B_T >= trace.min_B_T
    assert trace.mirror_ratio >= 1.0
    assert trace.end_R_m == pytest.approx(trace.forward.end_R_m)


def test_constant_vertical_field_hits_each_wall_and_reverse_is_opposite():
    field = ConstantField(B_Z_T=1.0)
    bounds = FieldLineBounds((0.2, 0.8), (-0.1, 0.1))
    tracer = FieldLineTracer(field, bounds, step_m=0.03, max_length_m=1.0)

    forward = tracer.trace(0.5, 0.0, direction="forward")
    backward = tracer.trace(0.5, 0.0, direction="backward")
    both = tracer.trace(0.5, 0.0, direction="both")

    assert forward.status == "wall_hit"
    assert forward.boundary == "Z_max"
    assert forward.end_Z_m == pytest.approx(0.1, abs=1.0e-10)
    assert forward.length_m == pytest.approx(0.1, abs=1.0e-10)
    assert backward.status == "wall_hit"
    assert backward.boundary == "Z_min"
    assert backward.end_Z_m == pytest.approx(-0.1, abs=1.0e-10)
    assert backward.length_m == pytest.approx(0.1, abs=1.0e-10)
    assert both.status == "wall_hit"
    assert both.hit_boundary
    assert both.boundary is None
    assert set(both.boundaries) == {"Z_min", "Z_max"}
    assert both.length_m == pytest.approx(0.2, abs=1.0e-10)


def test_polygonal_first_wall_reports_material_segment_hit():
    field = ConstantField(B_Z_T=1.0)
    bounds = FieldLineBounds(
        (0.1, 0.9),
        (-0.4, 0.4),
        (
            (0.2, -0.2),
            (0.8, -0.2),
            (0.7, 0.2),
            (0.3, 0.2),
        ),
    )
    tracer = FieldLineTracer(field, bounds, step_m=0.03, max_length_m=1.0)
    result = tracer.trace(0.5, 0.0, direction="forward")
    assert result.status == "wall_hit"
    assert result.end_Z_m == pytest.approx(0.2, abs=1.0e-9)
    assert result.boundary is not None and result.boundary.startswith("wall_segment_")


def test_geometry_wall_polygon_is_used_by_default():
    geometry = GeometryConfig(
        R0_m=0.5,
        minor_radius_m=0.2,
        R_bounds_m=(0.1, 0.9),
        Z_bounds_m=(-0.4, 0.4),
        wall_polygon_RZ_m=[
            (0.2, -0.2),
            (0.8, -0.2),
            (0.8, 0.2),
            (0.2, 0.2),
        ],
    )
    tracer = FieldLineTracer(ConstantField(B_Z_T=1.0), geometry=geometry, step_m=0.05)
    assert tracer.bounds.wall_polygon_RZ_m is not None
    assert not tracer.bounds.contains(0.5, 0.3)


def test_pure_toroidal_field_closes_after_one_toroidal_circumference():
    field = ConstantField(B_phi_T=1.0)
    tracer = FieldLineTracer(
        field,
        bounds=((0.1, 0.9), (-0.2, 0.2)),
        step_m=0.1,
        max_length_m=5.0,
        closure_tolerance_m=1.0e-4,
    )
    closed = tracer.trace(0.5, 0.0, direction="forward")
    timeout = tracer.trace(0.5, 0.0, direction="forward", max_length_m=2.0)

    assert closed.status == "closed"
    assert closed.closed
    assert closed.length_m == pytest.approx(2.0 * math.pi * 0.5, abs=0.11)
    assert timeout.status == "timeout"
    assert timeout.timed_out
    assert timeout.length_m == pytest.approx(2.0)


def test_composite_provider_can_be_traced_without_global_model_coupling():
    geometry = GeometryConfig(
        B_vertical_T=0.0,
        R_bounds_m=(0.15, 0.85),
        Z_bounds_m=(-0.40, 0.40),
        coil_components=[
            {"name": "weak_pf", "B_Z_T_per_A": 1.0e-9, "current_A": 1.0},
        ],
    )
    field = make_magnetic_field(geometry)
    trace = trace_field_line(
        field,
        geometry.R0_m + 0.10,
        0.0,
        Ip_A=100_000.0,
        direction="forward",
        step_m=0.01,
        max_length_m=0.5,
        closure_tolerance_m=0.005,
    )

    assert trace.provider_configuration.startswith("composite[")
    assert trace.length_m > 0.0
    assert trace.branches[0].n_steps > 0


def test_outside_and_zero_field_terminations_are_explicit():
    bounds = FieldLineBounds((0.2, 0.8), (-0.1, 0.1))
    outside = FieldLineTracer(ConstantField(B_Z_T=1.0), bounds).trace(0.1, 0.0, direction="forward")
    zero = FieldLineTracer(ConstantField(), bounds).trace(0.5, 0.0, direction="forward")
    assert outside.status == "outside_boundary"
    assert not outside.hit_boundary
    assert outside.length_m == 0.0
    assert zero.status == "zero_field"
    assert not zero.hit_boundary
    assert zero.length_m == 0.0
