"""Audited device geometry and experimental-waveform binding.

This module deliberately separates a drawing-derived topology record from a
physics-ready metric geometry.  A screenshot can establish coil names and
relative placement, but cannot establish metres, turns, conductor extent or
polarity.  ``build_experimental_magnetic_field`` refuses such incomplete
records instead of silently inventing those quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import GeometryConfig
from .experimental import EXL50UShot
from .physics.biot_savart import BiotSavartMagneticField, CircularLoop
from .physics.magnetic import FieldPoint, MagneticConfigurationState
from .waveform import linear_value


@dataclass(frozen=True)
class AxisymmetricCoilGeometry:
    name: str
    channel: str
    role: str
    center_R_m: float | None
    center_Z_m: float | None
    turns: int | None
    polarity: int | None
    cross_section_width_m: float | None = None
    cross_section_height_m: float | None = None
    relative_location: str = ""
    source: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "AxisymmetricCoilGeometry":
        return cls(
            name=str(payload["name"]),
            channel=str(payload["channel"]),
            role=str(payload.get("role", "PF")),
            center_R_m=_optional_float(payload.get("center_R_m")),
            center_Z_m=_optional_float(payload.get("center_Z_m")),
            turns=_optional_int(payload.get("turns")),
            polarity=_optional_int(payload.get("polarity")),
            cross_section_width_m=_optional_float(payload.get("cross_section_width_m")),
            cross_section_height_m=_optional_float(payload.get("cross_section_height_m")),
            relative_location=str(payload.get("relative_location", "")),
            source=str(payload.get("source", "")),
        )

    def missing_metric_fields(self) -> tuple[str, ...]:
        required = {
            "center_R_m": self.center_R_m,
            "center_Z_m": self.center_Z_m,
            "turns": self.turns,
            "polarity": self.polarity,
            "cross_section_width_m": self.cross_section_width_m,
            "cross_section_height_m": self.cross_section_height_m,
        }
        return tuple(name for name, value in required.items() if value is None)


@dataclass(frozen=True)
class DeviceGeometrySpec:
    device: str
    geometry_status: str
    R0_m: float
    minor_radius_m: float
    R_bounds_m: tuple[float, float]
    Z_bounds_m: tuple[float, float]
    wall_polygon_RZ_m: tuple[tuple[float, float], ...]
    coils: tuple[AxisymmetricCoilGeometry, ...]
    toroidal_field_channel: str = "BT"
    source_notes: tuple[str, ...] = ()

    def audit(self, shot: EXL50UShot | None = None) -> dict[str, object]:
        missing: list[str] = []
        if self.geometry_status != "metric_confirmed":
            missing.append("geometry_status must be 'metric_confirmed'")
        if len(self.wall_polygon_RZ_m) < 3:
            missing.append("measured first-wall/limiter polygon")
        names = [coil.name for coil in self.coils]
        if len(names) != len(set(names)):
            missing.append("unique coil names")
        for coil in self.coils:
            for field in coil.missing_metric_fields():
                missing.append(f"{coil.name}.{field}")
            if shot is not None and coil.channel not in shot.signals:
                missing.append(f"shot channel {coil.channel}")
        if shot is not None:
            if self.toroidal_field_channel not in shot.signals:
                missing.append(f"shot channel {self.toroidal_field_channel}")
            else:
                calibration = shot.signals[self.toroidal_field_channel].calibration
                if calibration.scale_to_physical is None:
                    missing.append(f"physical calibration for {self.toroidal_field_channel}")
            for coil in self.coils:
                signal = shot.signals.get(coil.channel)
                if signal is not None and signal.calibration.scale_to_physical is None:
                    missing.append(f"physical calibration for {coil.channel}")
        missing = sorted(set(missing))
        return {
            "device": self.device,
            "geometry_status": self.geometry_status,
            "physics_ready": not missing,
            "missing_requirements": missing,
            "coil_count": len(self.coils),
            "wall_vertex_count": len(self.wall_polygon_RZ_m),
            "equilibrium_dependency": "none; wall is a material boundary, not an LCFS",
            "source_notes": list(self.source_notes),
        }

    def geometry_config(self, B_t_T: float) -> GeometryConfig:
        return GeometryConfig(
            R0_m=self.R0_m,
            minor_radius_m=self.minor_radius_m,
            B_t_T=max(float(B_t_T), 1.0e-9),
            B_vertical_T=0.0,
            R_bounds_m=self.R_bounds_m,
            Z_bounds_m=self.Z_bounds_m,
            wall_polygon_RZ_m=list(self.wall_polygon_RZ_m) or None,
        )


class ExperimentalVacuumToroidalField:
    """Measured axis toroidal field with no invented plasma poloidal field."""

    configuration = "experimental_vacuum_toroidal"

    def __init__(
        self,
        geometry: GeometryConfig,
        B_axis_waveform_T: tuple[tuple[float, float], ...],
    ) -> None:
        if not B_axis_waveform_T:
            raise ValueError("B-axis waveform cannot be empty")
        self.geometry = geometry
        self.B_axis_waveform_T = B_axis_waveform_T

    def B_axis_T(self, t_s: float) -> float:
        return linear_value(self.B_axis_waveform_T, t_s, self.B_axis_waveform_T[0][1])

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0) -> FieldPoint:
        return FieldPoint(0.0, 0.0, self.B_axis_T(t_s) * self.geometry.R0_m / max(float(R_m), 1.0e-9))

    def flux_function(self, R_m, Z_m, Ip_A: float, t_s: float = 0.0):
        values = np.zeros(np.broadcast(np.asarray(R_m), np.asarray(Z_m)).shape, dtype=float)
        return float(values) if values.ndim == 0 else values

    def vertical_field(self, t_s: float = 0.0) -> float:
        return 0.0

    def configuration_state(self, Ip_A: float, t_s: float = 0.0) -> MagneticConfigurationState:
        return MagneticConfigurationState(configuration=self.configuration)

    def diagnostics(self, Ip_A: float, t_s: float = 0.0) -> dict[str, float | str]:
        return {
            "field_configuration": self.configuration,
            "field_provider": "measured_Bt_waveform_no_plasma_self_field",
            "B_t_axis_T": self.B_axis_T(t_s),
            "Bp_edge_provider_T": 0.0,
            "B_vertical_provider_T": 0.0,
            "B_vertical_topology_T": 0.0,
            "vertical_gradient_T_per_m": 0.0,
        }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("geometry values must be finite")
    return converted


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    converted = int(value)
    if converted != value:
        raise ValueError("integer geometry values must be exact integers")
    return converted


def load_device_geometry(path: str | Path) -> DeviceGeometrySpec:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    wall = tuple(tuple(float(value) for value in point) for point in payload.get("wall_polygon_RZ_m", []))
    return DeviceGeometrySpec(
        device=str(payload["device"]),
        geometry_status=str(payload.get("geometry_status", "schematic")),
        R0_m=float(payload["R0_m"]),
        minor_radius_m=float(payload["minor_radius_m"]),
        R_bounds_m=tuple(float(value) for value in payload["R_bounds_m"]),
        Z_bounds_m=tuple(float(value) for value in payload["Z_bounds_m"]),
        wall_polygon_RZ_m=wall,
        coils=tuple(AxisymmetricCoilGeometry.from_mapping(item) for item in payload.get("coils", [])),
        toroidal_field_channel=str(payload.get("toroidal_field_channel", "BT")),
        source_notes=tuple(str(note) for note in payload.get("source_notes", [])),
    )


def build_experimental_magnetic_field(
    geometry: DeviceGeometrySpec,
    shot: EXL50UShot,
) -> BiotSavartMagneticField:
    audit = geometry.audit(shot)
    if not audit["physics_ready"]:
        missing = ", ".join(audit["missing_requirements"])
        raise ValueError(f"device geometry is not physics-ready: {missing}")
    B_signal = shot.require(geometry.toroidal_field_channel)
    B_waveform = B_signal.waveform(physical=True)
    B_reference = float(np.median(B_signal.physical_values))
    config = geometry.geometry_config(B_reference)
    base = ExperimentalVacuumToroidalField(config, B_waveform)
    loops = []
    for coil in geometry.coils:
        signal = shot.require(coil.channel)
        polarity = int(coil.polarity)
        waveform = tuple((time, polarity * current) for time, current in signal.waveform(physical=True))
        loops.append(
            CircularLoop(
                name=coil.name,
                radius_m=float(coil.center_R_m),
                z0_m=float(coil.center_Z_m),
                turns=int(coil.turns),
                current_waveform_A=waveform,
            )
        )
    return BiotSavartMagneticField(config, loops, base_provider=base)


__all__ = [
    "AxisymmetricCoilGeometry",
    "DeviceGeometrySpec",
    "ExperimentalVacuumToroidalField",
    "load_device_geometry",
    "build_experimental_magnetic_field",
]
