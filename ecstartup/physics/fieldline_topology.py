"""Field-line based topology scans.

This module is an optional adapter between :mod:`fieldline` and the topology
contract used by the 0-D startup model.  It deliberately does not replace
``SurrogateTopologyDetector``: callers opt in by constructing
``FieldLineTopologyDetector`` and injecting it into their own workflow.

The scanner samples a polar grid in the poloidal cross-section.  A radial
sample is interpreted as ``rho = r/a`` by default and a poloidal sample as an
angle in radians.  The default area weights are the product of radial annulus
areas and angular Voronoi-cell widths.  Every trace, including failures, is
returned in a structured record so a coil singularity or an invalid provider
response cannot silently change a scan's denominator.

The implementation is intentionally provider-agnostic.  It only requires a
``MagneticFieldProvider`` and a ``FieldLineTracer``-like object with
``trace(...)``.  This makes it usable with the analytic, Biot--Savart and
future equilibrium/field-line backends without coupling any of them to the
global startup ODE.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from numbers import Integral
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..config import GeometryConfig
from .fieldline import FieldLineBounds, FieldLineTrace, FieldLineTracer
from .magnetic import MagneticFieldProvider, make_magnetic_field
from .topology import TopologyResult


InvalidPolicy = Literal["record", "raise"]
RadialCoordinateMode = Literal["fraction", "radius_m", "auto"]


class FieldLineTopologyError(RuntimeError):
    """Raised when a scan is configured to fail on an invalid trajectory."""


@dataclass(frozen=True)
class FieldLineScanConfig:
    """Configuration for :class:`FieldLineTopologyScanner`.

    ``radial_seeds`` and ``poloidal_seeds`` accept either a positive integer
    (which creates a uniform grid) or an explicit sequence.  Explicit radial
    values are normalized radii in ``[0, 1]`` unless
    ``radial_coordinate_mode="radius_m"`` is selected.  The aliases
    ``radial_fractions``, ``radial_seed_fractions`` and
    ``poloidal_angles_rad`` are provided for readable input files.

    ``seed_points`` can be supplied for a non-polar grid.  In that mode each
    point receives equal weight (or equal weight after ``area_weighted`` is
    disabled), because a general area measure cannot be inferred from an
    arbitrary point cloud.
    """

    radial_seeds: int | Sequence[float] = 5
    poloidal_seeds: int | Sequence[float] = 16
    radial_fractions: Sequence[float] | None = None
    radial_seed_fractions: Sequence[float] | None = None
    radial_radii_m: Sequence[float] | None = None
    radial_seed_radii_m: Sequence[float] | None = None
    poloidal_angles_rad: Sequence[float] | None = None
    poloidal_angles: Sequence[float] | None = None
    # Verbose aliases used by some parameter-scan notebooks.
    poloidal_seed_angles_rad: Sequence[float] | None = None
    seed_points: Sequence[Sequence[float]] | None = None
    area_weighted: bool = True
    # Descriptive alias accepted by callers constructing config from a mapping.
    area_weighting: bool | None = None
    radial_coordinate_mode: RadialCoordinateMode = "fraction"
    direction: str = "both"
    start_phi_rad: float = 0.0
    invalid_policy: InvalidPolicy = "record"
    count_timeout_as_valid: bool = True
    retain_traces: bool = True
    # Optional explicit geometry center.  If omitted, field.geometry is used.
    center_R_m: float | None = None
    center_Z_m: float | None = None
    minor_radius_m: float | None = None

    def __post_init__(self) -> None:
        if self.area_weighting is not None:
            object.__setattr__(self, "area_weighted", bool(self.area_weighting))
        radial_mode = str(self.radial_coordinate_mode).strip().lower()
        if radial_mode not in {"fraction", "radius_m", "auto"}:
            raise ValueError("radial_coordinate_mode 必须是 fraction、radius_m 或 auto")
        object.__setattr__(self, "radial_coordinate_mode", radial_mode)
        if self.invalid_policy not in {"record", "raise"}:
            raise ValueError("invalid_policy 必须是 record 或 raise")
        direction = str(self.direction).strip().lower()
        if direction not in {
            "forward",
            "backward",
            "both",
            "+",
            "-",
            "fwd",
            "back",
            "reverse",
            "positive",
            "negative",
            "bi",
            "bidirectional",
        }:
            raise ValueError("direction 必须是 forward、backward 或 both")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "area_weighted", bool(self.area_weighted))
        object.__setattr__(self, "count_timeout_as_valid", bool(self.count_timeout_as_valid))
        object.__setattr__(self, "retain_traces", bool(self.retain_traces))
        phi = float(self.start_phi_rad)
        if not math.isfinite(phi):
            raise ValueError("start_phi_rad 必须为有限数")
        object.__setattr__(self, "start_phi_rad", phi)
        for name in ("center_R_m", "center_Z_m", "minor_radius_m"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{name} 必须为有限数")
                if name == "minor_radius_m" and value <= 0.0:
                    raise ValueError("minor_radius_m 必须为正数")
                object.__setattr__(self, name, value)
        if self.seed_points is not None:
            points = []
            for point in self.seed_points:
                try:
                    if len(point) != 2:
                        raise ValueError
                    R, Z = float(point[0]), float(point[1])
                except (TypeError, ValueError, IndexError) as exc:
                    raise ValueError("seed_points 必须是有限的 (R_m, Z_m) 点列") from exc
                if not (math.isfinite(R) and math.isfinite(Z)):
                    raise ValueError("seed_points 必须是有限的 (R_m, Z_m) 点列")
                points.append((R, Z))
            if not points:
                raise ValueError("seed_points 不能为空")
            object.__setattr__(self, "seed_points", tuple(points))
        for name in (
            "radial_fractions",
            "radial_seed_fractions",
            "radial_radii_m",
            "radial_seed_radii_m",
            "poloidal_angles_rad",
            "poloidal_angles",
            "poloidal_seed_angles_rad",
        ):
            value = getattr(self, name)
            if value is not None:
                try:
                    values = tuple(float(item) for item in value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} 必须是有限数列") from exc
                if not values or any(not math.isfinite(item) for item in values):
                    raise ValueError(f"{name} 不能为空且必须只包含有限数")
                object.__setattr__(self, name, values)
        # Freeze explicit selectors as tuples so a caller cannot mutate a
        # scan configuration after it has been used to build weights.
        for name in ("radial_seeds", "poloidal_seeds"):
            selector = getattr(self, name)
            if not isinstance(selector, Integral) or isinstance(selector, bool):
                try:
                    selector = tuple(float(item) for item in selector)
                except TypeError as exc:
                    raise ValueError(f"{name} 必须是正整数或数列") from exc
                object.__setattr__(self, name, selector)
            else:
                selector = int(selector)
                object.__setattr__(self, name, selector)
            self._validate_seed_selector(selector, name)

    @staticmethod
    def _validate_seed_selector(value: int | Sequence[float], name: str) -> None:
        if isinstance(value, bool):
            raise ValueError(f"{name} 不能是布尔值")
        if isinstance(value, Integral) and not isinstance(value, bool):
            if value < 1:
                raise ValueError(f"{name} 的整数形式必须为正数")
            return
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} 必须是正整数或数列") from exc
        if not values:
            raise ValueError(f"{name} 数列不能为空")
        for item in values:
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"{name} 必须只包含有限数")

    @property
    def resolved_radial_selector(self) -> int | Sequence[float]:
        """Return the explicit radial alias with deterministic precedence."""

        if self.radial_radii_m is not None:
            return self.radial_radii_m
        if self.radial_seed_radii_m is not None:
            return self.radial_seed_radii_m
        if self.radial_seed_fractions is not None:
            return self.radial_seed_fractions
        if self.radial_fractions is not None:
            return self.radial_fractions
        return self.radial_seeds

    @property
    def resolved_poloidal_selector(self) -> int | Sequence[float]:
        if self.poloidal_angles_rad is not None:
            return self.poloidal_angles_rad
        if self.poloidal_seed_angles_rad is not None:
            return self.poloidal_seed_angles_rad
        if self.poloidal_angles is not None:
            return self.poloidal_angles
        return self.poloidal_seeds


@dataclass(frozen=True)
class FieldLineTopologySample:
    """One weighted seed and its field-line outcome."""

    index: int
    radial_fraction: float
    poloidal_angle_rad: float
    start_R_m: float
    start_Z_m: float
    weight: float
    status: str
    valid: bool
    closed: bool
    connection_length_m: float | None = None
    mirror_ratio: float | None = None
    min_B_T: float | None = None
    max_B_T: float | None = None
    error_kind: str | None = None
    error_message: str | None = None
    trace: FieldLineTrace | None = None

    @property
    def wall_hit(self) -> bool:
        return self.status == "wall_hit"

    @property
    def invalid(self) -> bool:
        return not self.valid

    def to_dict(self, *, include_trace: bool = False, include_path: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "index": self.index,
            "radial_fraction": self.radial_fraction,
            "poloidal_angle_rad": self.poloidal_angle_rad,
            "start_R_m": self.start_R_m,
            "start_Z_m": self.start_Z_m,
            "weight": self.weight,
            "status": self.status,
            "valid": self.valid,
            "invalid": self.invalid,
            "closed": self.closed,
            "wall_hit": self.wall_hit,
            "connection_length_m": self.connection_length_m,
            "mirror_ratio": self.mirror_ratio,
            "min_B_T": self.min_B_T,
            "max_B_T": self.max_B_T,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
        }
        if include_trace and self.trace is not None:
            result["trace"] = self.trace.to_dict(include_path=include_path)
        return result


# Shorter spelling for notebooks and backwards-compatible experiments.
FieldLineScanSample = FieldLineTopologySample


@dataclass(frozen=True)
class FieldLineTopologyScanResult:
    """Aggregated result of a cross-section field-line scan.

    ``closed_fraction`` is conditional on the area represented by valid
    trajectories.  ``closed_area_fraction`` retains the denominator of the
    complete requested grid, while ``valid_fraction`` reports how much of that
    area was successfully traced.  This distinction is important near an
    ideal filament coil, where evaluating exactly on the wire is undefined.
    """

    provider_configuration: str
    Ip_A: float
    t_s: float
    center_R_m: float
    center_Z_m: float
    minor_radius_m: float
    total_sample_count: int
    valid_sample_count: int
    invalid_sample_count: int
    closed_sample_count: int
    total_weight: float
    valid_weight: float
    closed_weight: float
    valid_fraction: float
    closed_fraction: float
    closed_area_fraction: float
    mean_connection_length_m: float
    mean_closed_connection_length_m: float
    mean_mirror_ratio: float
    mean_closed_mirror_ratio: float
    min_connection_length_m: float
    max_connection_length_m: float
    effective_sample_count: float
    samples: tuple[FieldLineTopologySample, ...]
    status_counts: Mapping[str, int]
    field_diagnostics: Mapping[str, Any]

    @property
    def n_samples(self) -> int:
        return self.total_sample_count

    @property
    def n_valid(self) -> int:
        return self.valid_sample_count

    @property
    def n_invalid(self) -> int:
        return self.invalid_sample_count

    @property
    def n_closed(self) -> int:
        return self.closed_sample_count

    @property
    def singular_sample_count(self) -> int:
        return int(self.status_counts.get("singular_field", 0))

    @property
    def invalid_trace_count(self) -> int:
        return int(
            sum(
                count
                for status, count in self.status_counts.items()
                if status in {"invalid_trace", "trace_error", "provider_error"}
            )
        )

    @property
    def wall_hit_sample_count(self) -> int:
        return int(self.status_counts.get("wall_hit", 0))

    @property
    def timeout_sample_count(self) -> int:
        return int(self.status_counts.get("timeout", 0))

    @property
    def average_connection_length_m(self) -> float:
        return self.mean_connection_length_m

    @property
    def average_mirror_ratio(self) -> float:
        return self.mean_mirror_ratio

    @property
    def average_connection_length(self) -> float:
        """Unit-explicit alias retained for compact scan tables."""

        return self.mean_connection_length_m

    @property
    def average_mirror_ratio_value(self) -> float:
        return self.mean_mirror_ratio

    @property
    def n_effective(self) -> float:
        return self.effective_sample_count

    @property
    def candidate_radius_m(self) -> float:
        """Equivalent circular radius for the closed requested area."""

        return self.minor_radius_m * math.sqrt(max(0.0, min(1.0, self.closed_area_fraction)))

    def to_dict(
        self,
        *,
        include_samples: bool = True,
        include_traces: bool = False,
        include_paths: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "provider_configuration": self.provider_configuration,
            "Ip_A": self.Ip_A,
            "t_s": self.t_s,
            "center_R_m": self.center_R_m,
            "center_Z_m": self.center_Z_m,
            "minor_radius_m": self.minor_radius_m,
            "total_sample_count": self.total_sample_count,
            "valid_sample_count": self.valid_sample_count,
            "invalid_sample_count": self.invalid_sample_count,
            "closed_sample_count": self.closed_sample_count,
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "n_closed": self.n_closed,
            "singular_sample_count": self.singular_sample_count,
            "invalid_trace_count": self.invalid_trace_count,
            "wall_hit_sample_count": self.wall_hit_sample_count,
            "timeout_sample_count": self.timeout_sample_count,
            "total_weight": self.total_weight,
            "valid_weight": self.valid_weight,
            "closed_weight": self.closed_weight,
            "valid_fraction": self.valid_fraction,
            "closed_fraction": self.closed_fraction,
            "closed_area_fraction": self.closed_area_fraction,
            "candidate_radius_m": self.candidate_radius_m,
            "mean_connection_length_m": self.mean_connection_length_m,
            "average_connection_length_m": self.average_connection_length_m,
            "mean_closed_connection_length_m": self.mean_closed_connection_length_m,
            "mean_mirror_ratio": self.mean_mirror_ratio,
            "average_mirror_ratio": self.average_mirror_ratio,
            "mean_closed_mirror_ratio": self.mean_closed_mirror_ratio,
            "min_connection_length_m": self.min_connection_length_m,
            "max_connection_length_m": self.max_connection_length_m,
            "effective_sample_count": self.effective_sample_count,
            "n_effective": self.n_effective,
            "status_counts": dict(self.status_counts),
            "field_diagnostics": dict(self.field_diagnostics),
        }
        if include_samples:
            result["samples"] = [
                sample.to_dict(include_trace=include_traces, include_path=include_paths)
                for sample in self.samples
            ]
        return result

    def as_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias used by the comparison/output helpers."""

        return self.to_dict(**kwargs)


@dataclass(frozen=True)
class _Seed:
    radial_fraction: float
    poloidal_angle_rad: float
    R_m: float
    Z_m: float
    weight: float


class FieldLineTopologyScanner:
    """Scan a magnetic cross-section by tracing configurable seed points."""

    def __init__(
        self,
        field: MagneticFieldProvider,
        tracer: FieldLineTracer | Any | None = None,
        config: FieldLineScanConfig | Mapping[str, Any] | None = None,
        *,
        scan_config: FieldLineScanConfig | Mapping[str, Any] | None = None,
        geometry: GeometryConfig | None = None,
        radial_seeds: int | Sequence[float] | None = None,
        poloidal_seeds: int | Sequence[float] | None = None,
        radial_fractions: Sequence[float] | None = None,
        radial_seed_fractions: Sequence[float] | None = None,
        poloidal_angles_rad: Sequence[float] | None = None,
        poloidal_seed_angles_rad: Sequence[float] | None = None,
        radial_radii_m: Sequence[float] | None = None,
        radial_seed_radii_m: Sequence[float] | None = None,
        area_weighted: bool | None = None,
        direction: str | None = None,
        invalid_policy: InvalidPolicy | None = None,
    ) -> None:
        if field is None or not callable(getattr(field, "evaluate", None)):
            raise TypeError("field 必须实现 MagneticFieldProvider.evaluate")
        if config is not None and scan_config is not None:
            raise ValueError("config 与 scan_config 不能同时提供")
        selected = config if config is not None else scan_config
        if selected is None:
            scan = FieldLineScanConfig()
        elif isinstance(selected, FieldLineScanConfig):
            scan = selected
        elif isinstance(selected, Mapping):
            scan = FieldLineScanConfig(**dict(selected))
        else:
            raise TypeError("config 必须是 FieldLineScanConfig 或映射")
        overrides: dict[str, Any] = {}
        for name, value in (
            ("radial_seeds", radial_seeds),
            ("poloidal_seeds", poloidal_seeds),
            ("radial_fractions", radial_fractions),
            ("radial_seed_fractions", radial_seed_fractions),
            ("poloidal_angles_rad", poloidal_angles_rad),
            ("poloidal_seed_angles_rad", poloidal_seed_angles_rad),
            ("radial_radii_m", radial_radii_m),
            ("radial_seed_radii_m", radial_seed_radii_m),
            ("area_weighted", area_weighted),
            ("direction", direction),
            ("invalid_policy", invalid_policy),
        ):
            if value is not None:
                overrides[name] = value
        if overrides:
            scan = replace(scan, **overrides)
        self.field = field
        selected_geometry = geometry or getattr(field, "geometry", None)
        if isinstance(selected_geometry, Mapping):
            selected_geometry = GeometryConfig.model_validate(dict(selected_geometry))
        self.geometry = selected_geometry
        self.tracer = tracer if tracer is not None else self._make_tracer()
        if not callable(getattr(self.tracer, "trace", None)):
            raise TypeError("tracer 必须提供 trace(...) 方法")
        self.config = scan
        self.center_R_m, self.center_Z_m, self.minor_radius_m = self._resolve_geometry(scan)
        self.last_result: FieldLineTopologyScanResult | None = None

    def _make_tracer(self) -> FieldLineTracer:
        if self.geometry is not None:
            return FieldLineTracer(self.field, bounds=self.geometry)
        bounds = getattr(self.field, "bounds", None)
        if bounds is None:
            raise ValueError("未提供 tracer 时，field 必须有 geometry；否则请显式传入 tracer")
        return FieldLineTracer(self.field, bounds=bounds)

    def _resolve_geometry(self, config: FieldLineScanConfig) -> tuple[float, float, float]:
        geometry = self.geometry
        bounds = getattr(self.tracer, "bounds", None)
        if isinstance(bounds, FieldLineBounds):
            r_bounds, z_bounds = bounds.R_bounds_m, bounds.Z_bounds_m
        elif bounds is not None:
            try:
                coerced = FieldLineBounds.coerce(bounds)
                r_bounds, z_bounds = coerced.R_bounds_m, coerced.Z_bounds_m
            except (TypeError, ValueError):
                r_bounds = z_bounds = None
        else:
            r_bounds = z_bounds = None
        R0 = config.center_R_m
        if R0 is None and geometry is not None:
            R0 = float(geometry.R0_m)
        if R0 is None and r_bounds is not None:
            R0 = 0.5 * (r_bounds[0] + r_bounds[1])
        if R0 is None:
            raise ValueError("无法确定扫描中心 R0_m；请提供 geometry 或 config.center_R_m")
        Z0 = config.center_Z_m
        if Z0 is None:
            # GeometryConfig uses Z=0 for its circular center.  A custom
            # tracer can still request a shifted center explicitly.
            Z0 = 0.0 if geometry is not None else (0.5 * (z_bounds[0] + z_bounds[1]) if z_bounds else 0.0)
        radius = config.minor_radius_m
        if radius is None and geometry is not None:
            radius = float(geometry.minor_radius_m)
        if radius is None and r_bounds is not None and z_bounds is not None:
            radius = min(0.5 * (r_bounds[1] - r_bounds[0]), 0.5 * (z_bounds[1] - z_bounds[0]))
        if radius is None or radius <= 0.0:
            raise ValueError("无法确定扫描 minor_radius_m；请提供 geometry 或 config.minor_radius_m")
        return float(R0), float(Z0), float(radius)

    @staticmethod
    def _deduplicate(values: Iterable[float], *, period: float | None = None) -> tuple[float, ...]:
        output: list[float] = []
        tol = 1.0e-12
        for raw in values:
            value = float(raw)
            if period is not None:
                value = value % period
            if not any(abs(value - old) <= tol for old in output):
                output.append(value)
        return tuple(output)

    def _radial_fractions(self) -> tuple[float, ...]:
        selector = self.config.resolved_radial_selector
        if isinstance(selector, Integral) and not isinstance(selector, bool):
            values = tuple((index + 0.5) / selector for index in range(selector))
        else:
            values = tuple(float(value) for value in selector)
            mode = self.config.radial_coordinate_mode
            if self.config.radial_radii_m is not None or self.config.radial_seed_radii_m is not None:
                mode = "radius_m"
            if mode == "radius_m" or (mode == "auto" and any(abs(value) > 1.0 for value in values)):
                values = tuple(value / self.minor_radius_m for value in values)
        if any(not math.isfinite(value) or value < -1.0e-12 or value > 1.0 + 1.0e-12 for value in values):
            raise ValueError("径向种子必须位于 [0, 1]（或 radius_m 模式下位于 [0, minor_radius_m]）")
        values = tuple(min(1.0, max(0.0, value)) for value in values)
        values = self._deduplicate(values)
        if not values:
            raise ValueError("径向种子不能为空")
        return values

    def _poloidal_angles(self) -> tuple[float, ...]:
        selector = self.config.resolved_poloidal_selector
        if isinstance(selector, Integral) and not isinstance(selector, bool):
            values = tuple(2.0 * math.pi * index / selector for index in range(selector))
        else:
            values = tuple(float(value) for value in selector)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("极向角种子必须为有限数")
        values = self._deduplicate(values, period=2.0 * math.pi)
        if not values:
            raise ValueError("极向角种子不能为空")
        return values

    @staticmethod
    def _angular_weights(angles: Sequence[float]) -> dict[float, float]:
        """Return circular Voronoi widths normalized to one."""

        if len(angles) == 1:
            return {float(angles[0]): 1.0}
        ordered = sorted(float(value) for value in angles)
        period = 2.0 * math.pi
        widths: dict[float, float] = {}
        for index, angle in enumerate(ordered):
            previous = ordered[index - 1] if index else ordered[-1] - period
            following = ordered[index + 1] if index + 1 < len(ordered) else ordered[0] + period
            width = 0.5 * (angle - previous) + 0.5 * (following - angle)
            widths[angle] = width / period
        return widths

    @staticmethod
    def _radial_weights(fractions: Sequence[float]) -> dict[float, float]:
        ordered = sorted(float(value) for value in fractions)
        if len(ordered) == 1:
            return {ordered[0]: 1.0}
        edges = [0.0]
        edges.extend(0.5 * (ordered[index] + ordered[index + 1]) for index in range(len(ordered) - 1))
        edges.append(1.0)
        weights: dict[float, float] = {}
        for index, value in enumerate(ordered):
            weights[value] = max(0.0, edges[index + 1] ** 2 - edges[index] ** 2)
        total = sum(weights.values())
        if total <= 0.0:
            return {value: 1.0 / len(ordered) for value in ordered}
        return {value: weight / total for value, weight in weights.items()}

    def _seeds(self) -> tuple[_Seed, ...]:
        if self.config.seed_points is not None:
            points = tuple(self.config.seed_points)
            equal = 1.0 / len(points)
            output = []
            for R, Z in points:
                rho = math.hypot(float(R) - self.center_R_m, float(Z) - self.center_Z_m) / self.minor_radius_m
                angle = math.atan2(float(Z) - self.center_Z_m, float(R) - self.center_R_m) % (2.0 * math.pi)
                output.append(_Seed(rho, angle, float(R), float(Z), equal))
            return tuple(output)

        fractions = self._radial_fractions()
        angles = self._poloidal_angles()
        radial_weights = self._radial_weights(fractions) if self.config.area_weighted else {value: 1.0 for value in fractions}
        angular_weights = self._angular_weights(angles) if self.config.area_weighted else {value: 1.0 / len(angles) for value in angles}
        raw: list[_Seed] = []
        for rho in fractions:
            # All polar angles coincide at the axis.  One representative avoids
            # artificial multiplicity while preserving the full annulus weight.
            angle_values = (0.0,) if rho <= 1.0e-14 else angles
            angle_norm = 1.0 if rho <= 1.0e-14 else None
            for angle in angle_values:
                R = self.center_R_m + self.minor_radius_m * rho * math.cos(angle)
                Z = self.center_Z_m + self.minor_radius_m * rho * math.sin(angle)
                angular_weight = angle_norm if angle_norm is not None else angular_weights[angle]
                raw.append(_Seed(rho, angle, R, Z, radial_weights[rho] * angular_weight))
        total = sum(seed.weight for seed in raw)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("种子面积权重无效")
        return tuple(replace(seed, weight=seed.weight / total) for seed in raw)

    @staticmethod
    def _error_kind(error: BaseException) -> str:
        text = f"{type(error).__name__}: {error}".lower()
        singular_words = ("singular", "wire", "filament", "发散", "细丝", "除零", "zero division")
        if isinstance(error, (ZeroDivisionError, FloatingPointError)) or any(word in text for word in singular_words):
            return "singular_field"
        return "trace_error"

    def _invoke_trace(self, seed: _Seed, Ip_A: float, t_s: float) -> Any:
        """Call both the native tracer and lightweight third-party tracers."""

        try:
            return self.tracer.trace(
                seed.R_m,
                seed.Z_m,
                start_phi_rad=self.config.start_phi_rad,
                Ip_A=Ip_A,
                t_s=t_s,
                direction=self.config.direction,
            )
        except TypeError as error:
            message = str(error).lower()
            if not any(token in message for token in ("unexpected", "keyword", "positional", "argument")):
                raise
            # A minimal adapter often exposes only trace(R, Z, Ip_A, t_s,
            # direction).  Retry without the optional toroidal angle.
            return self.tracer.trace(seed.R_m, seed.Z_m, Ip_A=Ip_A, t_s=t_s, direction=self.config.direction)

    @staticmethod
    def _extract_trace(trace: Any) -> tuple[str, bool, float, float, float, float]:
        if trace is None:
            raise ValueError("tracer 返回 None")
        # Accept a dictionary returned by a lightweight external adapter in
        # addition to the native FieldLineTrace dataclass.
        if isinstance(trace, Mapping):
            getter = trace.get
        else:
            getter = lambda key, default=None: getattr(trace, key, default)
        raw_status = getter("status")
        if raw_status is None or not str(raw_status).strip():
            raise ValueError("trace 缺少 status")
        status = str(raw_status).strip()
        closed = bool(getter("closed", status == "closed"))
        try:
            raw_length = getter("length_m", None)
            if raw_length is None:
                raw_length = getter("connection_length_m")
            length = float(raw_length)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("trace 缺少有限 length_m") from error
        mirror = float(getter("mirror_ratio", 1.0))
        min_B = float(getter("min_B_T", 0.0))
        max_B = float(getter("max_B_T", 0.0))
        if not math.isfinite(length) or length < 0.0:
            raise ValueError("trace.length_m 必须为有限非负数")
        if (not math.isfinite(mirror) and mirror != math.inf) or mirror < 0.0:
            raise ValueError("trace.mirror_ratio 必须为有限非负数或 inf")
        if not all(math.isfinite(value) and value >= 0.0 for value in (min_B, max_B)):
            raise ValueError("trace 的 B 诊断必须为有限非负数")
        if max_B + 1.0e-12 < min_B:
            raise ValueError("trace.max_B_T 不能小于 min_B_T")
        # Check endpoints when the rich FieldLineTrace object provides them.
        for name in ("end_R_m", "end_Z_m", "end_phi_rad"):
            value = getter(name, None)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"trace.{name} 不是有限数")
        for name in ("path_s_m", "path_R_m", "path_Z_m", "path_phi_rad"):
            values = getter(name, None)
            if values is not None:
                try:
                    if any(not math.isfinite(float(value)) for value in values):
                        raise ValueError(f"trace.{name} 包含非有限数")
                except TypeError as error:
                    raise ValueError(f"trace.{name} 不是数列") from error
        return status, closed, length, mirror, min_B, max_B

    def _invalid_sample(
        self,
        seed: _Seed,
        index: int,
        kind: str,
        message: str,
    ) -> FieldLineTopologySample:
        if self.config.invalid_policy == "raise":
            raise FieldLineTopologyError(f"field-line seed {index} ({seed.R_m:.6g}, {seed.Z_m:.6g}) 无效: {message}")
        return FieldLineTopologySample(
            index=index,
            radial_fraction=seed.radial_fraction,
            poloidal_angle_rad=seed.poloidal_angle_rad,
            start_R_m=seed.R_m,
            start_Z_m=seed.Z_m,
            weight=seed.weight,
            status=kind,
            valid=False,
            closed=False,
            error_kind=kind,
            error_message=message,
        )

    def _sample(self, seed: _Seed, index: int, Ip_A: float, t_s: float) -> FieldLineTopologySample:
        try:
            trace = self._invoke_trace(seed, Ip_A, t_s)
            status, closed, length, mirror, min_B, max_B = self._extract_trace(trace)
        except Exception as error:
            kind = self._error_kind(error)
            return self._invalid_sample(seed, index, kind, str(error))
        # These statuses indicate that no physical trajectory was obtained.
        # ``timeout`` remains valid by default, but can be excluded when a
        # scan requires a completed closure/wall event.
        invalid_statuses = {"outside_boundary", "zero_field", "invalid", "error", "provider_error", "invalid_trace"}
        valid = status not in invalid_statuses
        # A custom tracer may use a different spelling for a failure.  Treat
        # an unknown non-closed status as invalid rather than allowing a
        # malformed result to enter the weighted averages.
        known_statuses = {"closed", "wall_hit", "timeout"}
        if status not in known_statuses and not closed:
            valid = False
        if status == "timeout" and not self.config.count_timeout_as_valid:
            valid = False
        if not valid:
            return self._invalid_sample(seed, index, status, f"tracer status={status}")
        retained = trace if self.config.retain_traces and isinstance(trace, FieldLineTrace) else None
        return FieldLineTopologySample(
            index=index,
            radial_fraction=seed.radial_fraction,
            poloidal_angle_rad=seed.poloidal_angle_rad,
            start_R_m=seed.R_m,
            start_Z_m=seed.Z_m,
            weight=seed.weight,
            status=status,
            valid=True,
            closed=closed,
            connection_length_m=length,
            mirror_ratio=mirror,
            min_B_T=min_B,
            max_B_T=max_B,
            trace=retained,
        )

    @staticmethod
    def _weighted_mean(samples: Sequence[FieldLineTopologySample], attr: str) -> float:
        selected = [sample for sample in samples if sample.valid and getattr(sample, attr) is not None]
        if not selected:
            return 0.0
        if any(getattr(sample, attr) == math.inf and sample.weight > 0.0 for sample in selected):
            return math.inf
        denominator = sum(sample.weight for sample in selected)
        if denominator <= 0.0:
            return 0.0
        return sum(sample.weight * float(getattr(sample, attr)) for sample in selected) / denominator

    @staticmethod
    def _safe_diagnostics(field: Any, Ip_A: float, t_s: float) -> dict[str, Any]:
        method = getattr(field, "diagnostics", None)
        if not callable(method):
            return {}
        try:
            data = method(Ip_A, t_s)
        except Exception as error:
            return {"diagnostics_error": f"{type(error).__name__}: {error}"}
        if not isinstance(data, Mapping):
            return {"diagnostics_error": "field.diagnostics 未返回映射"}
        return dict(data)

    def scan(self, Ip_A: float, t_s: float = 0.0) -> FieldLineTopologyScanResult:
        ip = float(Ip_A)
        time = float(t_s)
        if not (math.isfinite(ip) and math.isfinite(time)):
            raise ValueError("Ip_A 和 t_s 必须为有限数")
        seeds = self._seeds()
        samples = tuple(self._sample(seed, index, ip, time) for index, seed in enumerate(seeds))
        total_weight = sum(sample.weight for sample in samples)
        valid = tuple(sample for sample in samples if sample.valid)
        closed = tuple(sample for sample in valid if sample.closed)
        valid_weight = sum(sample.weight for sample in valid)
        closed_weight = sum(sample.weight for sample in closed)
        closed_fraction = closed_weight / valid_weight if valid_weight > 0.0 else 0.0
        closed_area_fraction = closed_weight / total_weight if total_weight > 0.0 else 0.0
        valid_fraction = valid_weight / total_weight if total_weight > 0.0 else 0.0
        finite_lengths = [float(sample.connection_length_m) for sample in valid if sample.connection_length_m is not None]
        min_length = min(finite_lengths) if finite_lengths else 0.0
        max_length = max(finite_lengths) if finite_lengths else 0.0
        sum_weight_sq = sum(sample.weight * sample.weight for sample in valid)
        effective = valid_weight * valid_weight / sum_weight_sq if sum_weight_sq > 0.0 else 0.0
        result = FieldLineTopologyScanResult(
            provider_configuration=str(getattr(self.field, "configuration", "custom")),
            Ip_A=ip,
            t_s=time,
            center_R_m=self.center_R_m,
            center_Z_m=self.center_Z_m,
            minor_radius_m=self.minor_radius_m,
            total_sample_count=len(samples),
            valid_sample_count=len(valid),
            invalid_sample_count=len(samples) - len(valid),
            closed_sample_count=len(closed),
            total_weight=total_weight,
            valid_weight=valid_weight,
            closed_weight=closed_weight,
            valid_fraction=valid_fraction,
            closed_fraction=closed_fraction,
            closed_area_fraction=closed_area_fraction,
            mean_connection_length_m=self._weighted_mean(samples, "connection_length_m"),
            mean_closed_connection_length_m=self._weighted_mean(closed, "connection_length_m"),
            mean_mirror_ratio=self._weighted_mean(samples, "mirror_ratio"),
            mean_closed_mirror_ratio=self._weighted_mean(closed, "mirror_ratio"),
            min_connection_length_m=min_length,
            max_connection_length_m=max_length,
            effective_sample_count=effective,
            samples=samples,
            status_counts=dict(Counter(sample.status for sample in samples)),
            field_diagnostics=self._safe_diagnostics(self.field, ip, time),
        )
        self.last_result = result
        return result


class FieldLineTopologyDetector:
    """TopologyProvider adapter backed by a field-line scan.

    Construct with an existing scanner, or pass a magnetic provider and tracer
    directly.  ``detect`` returns the project's existing ``TopologyResult``;
    the complete scan (including mirror statistics and invalid samples) is
    available as ``last_scan``.
    """

    def __init__(
        self,
        scanner: FieldLineTopologyScanner | MagneticFieldProvider | GeometryConfig | None = None,
        tracer: FieldLineTracer | Any | None = None,
        *,
        field: MagneticFieldProvider | None = None,
        geometry: GeometryConfig | None = None,
        config: FieldLineScanConfig | Mapping[str, Any] | None = None,
        scan_config: FieldLineScanConfig | Mapping[str, Any] | None = None,
        closed_fraction_threshold: float = 0.05,
        penetration_field_T: float | None = None,
        radial_seeds: int | Sequence[float] | None = None,
        poloidal_seeds: int | Sequence[float] | None = None,
        radial_fractions: Sequence[float] | None = None,
        radial_seed_fractions: Sequence[float] | None = None,
        poloidal_angles_rad: Sequence[float] | None = None,
        poloidal_seed_angles_rad: Sequence[float] | None = None,
        radial_radii_m: Sequence[float] | None = None,
        radial_seed_radii_m: Sequence[float] | None = None,
        area_weighted: bool | None = None,
    ) -> None:
        if isinstance(scanner, FieldLineTopologyScanner):
            if tracer is not None or any(item is not None for item in (field, geometry, config, scan_config, radial_seeds, poloidal_seeds, radial_fractions, radial_seed_fractions, poloidal_angles_rad, poloidal_seed_angles_rad, radial_radii_m, radial_seed_radii_m, area_weighted)):
                raise ValueError("传入 scanner 后不能重复指定 field/geometry/scan 配置")
            self.scanner = scanner
        else:
            selected_field = field
            selected_geometry = geometry
            if isinstance(scanner, GeometryConfig):
                if selected_geometry is not None:
                    raise ValueError("geometry 被重复指定")
                selected_geometry = scanner
                scanner = None
            elif scanner is not None:
                if selected_field is not None:
                    raise ValueError("field 被重复指定")
                selected_field = scanner  # type: ignore[assignment]
            if selected_field is None:
                if selected_geometry is None:
                    raise ValueError("请提供 scanner、field 或 geometry")
                selected_field = make_magnetic_field(selected_geometry)
            self.scanner = FieldLineTopologyScanner(
                selected_field,
                tracer,
                config=config,
                scan_config=scan_config,
                geometry=selected_geometry,
                radial_seeds=radial_seeds,
                poloidal_seeds=poloidal_seeds,
                radial_fractions=radial_fractions,
                radial_seed_fractions=radial_seed_fractions,
                poloidal_angles_rad=poloidal_angles_rad,
                poloidal_seed_angles_rad=poloidal_seed_angles_rad,
                radial_radii_m=radial_radii_m,
                radial_seed_radii_m=radial_seed_radii_m,
                area_weighted=area_weighted,
            )
        threshold = float(closed_fraction_threshold)
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("closed_fraction_threshold 必须位于 [0, 1]")
        self.closed_fraction_threshold = threshold
        if penetration_field_T is None:
            geometry = self.scanner.geometry
            penetration_field_T = float(getattr(geometry, "B_penetration_T", 0.0))
        penetration_field_T = float(penetration_field_T)
        if not math.isfinite(penetration_field_T) or penetration_field_T < 0.0:
            raise ValueError("penetration_field_T 必须为有限非负数")
        self.penetration_field_T = penetration_field_T
        self.last_scan: FieldLineTopologyScanResult | None = None

    def scan(self, Ip_A: float, t_s: float = 0.0) -> FieldLineTopologyScanResult:
        result = self.scanner.scan(Ip_A, t_s)
        self.last_scan = result
        return result

    def topology_from_scan(self, scan: FieldLineTopologyScanResult) -> TopologyResult:
        """Map an already-computed scan to ``TopologyResult``.

        Keeping this conversion separate lets command-line and batch callers
        attach the compatibility result without tracing the same seeds twice.
        ``detect`` remains the normal one-call ``TopologyProvider`` entry.
        """

        if not isinstance(scan, FieldLineTopologyScanResult):
            raise TypeError("scan 必须是 FieldLineTopologyScanResult")
        self.last_scan = scan
        bp_edge, threshold = self._edge_fields(scan.Ip_A, scan.t_s, scan)
        has_closed = scan.valid_sample_count > 0 and scan.closed_fraction >= self.closed_fraction_threshold
        state_method = getattr(self.scanner.field, "configuration_state", None)
        try:
            state = state_method(float(scan.Ip_A), float(scan.t_s)) if callable(state_method) else None
        except Exception:
            state = None
        configuration = str(getattr(self.scanner.field, "configuration", scan.provider_configuration))
        if state is not None:
            configuration = str(getattr(state, "configuration", configuration))
        confidence = scan.valid_fraction * min(1.0, math.sqrt(max(scan.effective_sample_count, 0.0)) / 4.0)
        confidence = min(1.0, max(0.0, confidence))
        if scan.valid_sample_count == 0:
            reason = "没有可用于拓扑判定的有效场线；请检查线圈奇异点、场强下限和种子范围"
        elif has_closed:
            reason = (
                f"场线扫描在有效面积中得到闭合比例 {scan.closed_fraction:.4g} "
                f"（{scan.closed_sample_count}/{scan.valid_sample_count} 个有效种子）"
            )
        else:
            reason = (
                f"场线扫描的闭合比例 {scan.closed_fraction:.4g} 未达到阈值 "
                f"{self.closed_fraction_threshold:.4g}"
            )
        connection = scan.mean_connection_length_m
        return TopologyResult(
            closed_fraction=scan.closed_fraction,
            candidate_radius_m=scan.candidate_radius_m,
            Bp_edge_T=bp_edge,
            threshold_T=threshold,
            has_closed_surface=has_closed,
            connection_length_m=connection,
            method=f"{configuration}_fieldline_scan",
            confidence=confidence,
            reason=reason,
            field_configuration=configuration,
            mirror_ratio=float(getattr(state, "mirror_ratio", scan.mean_mirror_ratio)) if state is not None else scan.mean_mirror_ratio,
            trapped_fraction=float(getattr(state, "trapped_fraction", 0.0)) if state is not None else 0.0,
            decay_index=float(getattr(state, "decay_index", 0.0)) if state is not None else 0.0,
            connection_length_factor=float(getattr(state, "connection_length_factor", 1.0)) if state is not None else 1.0,
            loss_multiplier=float(getattr(state, "loss_multiplier", 1.0)) if state is not None else 1.0,
        )

    def _edge_fields(self, Ip_A: float, t_s: float, scan: FieldLineTopologyScanResult) -> tuple[float, float]:
        diagnostics = scan.field_diagnostics
        try:
            bp = float(diagnostics.get("Bp_edge_provider_T", math.nan))
        except (TypeError, ValueError):
            bp = math.nan
        if not math.isfinite(bp):
            try:
                point = self.scanner.field.evaluate(
                    self.scanner.center_R_m + self.scanner.minor_radius_m,
                    self.scanner.center_Z_m,
                    Ip_A,
                    t_s,
                )
                bp = math.hypot(float(point.B_R_T), float(point.B_Z_T))
            except Exception:
                bp = 0.0
        vertical_value = diagnostics.get("B_vertical_topology_T", diagnostics.get("B_vertical_provider_T", math.nan))
        try:
            vertical = abs(float(vertical_value))
        except (TypeError, ValueError):
            vertical = 0.0
        return max(0.0, bp), vertical + self.penetration_field_T

    def detect(self, Ip_A: float, t_s: float = 0.0) -> TopologyResult:
        return self.topology_from_scan(self.scan(Ip_A, t_s))


# Names used in early notebooks and by downstream callers.
FieldLineTopologyScanAdapter = FieldLineTopologyDetector
FieldLineScanTopologyDetector = FieldLineTopologyDetector
FieldLineTopologyScanConfig = FieldLineScanConfig
FieldLineScanResult = FieldLineTopologyScanResult
FieldLineTopologyAdapter = FieldLineTopologyDetector


__all__ = [
    "FieldLineTopologyError",
    "FieldLineScanConfig",
    "FieldLineTopologySample",
    "FieldLineScanSample",
    "FieldLineTopologyScanResult",
    "FieldLineTopologyScanner",
    "FieldLineTopologyDetector",
    "FieldLineTopologyScanAdapter",
    "FieldLineScanTopologyDetector",
    "FieldLineTopologyScanConfig",
    "FieldLineScanResult",
    "FieldLineTopologyAdapter",
]
