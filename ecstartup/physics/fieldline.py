r"""轴对称磁场线追踪与开场线连接长度。

本模块只依赖 ``MagneticFieldProvider.evaluate``，因此可以用于圆截面、
TPC/VFC surrogate、``CompositeMagneticField`` 以及将来的线圈/GS provider，
而不需要把场线积分耦合进 0D 启动方程。磁场分量采用柱坐标
``(B_R, B_Z, B_phi)``，积分参数是物理弧长 ``s``：

.. math::

   dR/ds=B_R/|B|,\quad dZ/ds=B_Z/|B|,\quad
   d\\phi/ds=B_\\phi/(R|B|).

``direction="both"`` 从同一点分别沿 ``+B`` 和 ``-B`` 积分。闭合判据
针对 ``(R,Z)`` 极向投影（适合判断是否回到同一磁通面），不是要求三维
螺旋线在整数圈后精确回到同一 ``phi``。边界支持矩形计算域或显式
的轴对称第一壁/limiter 多边形；后者只是材料边界，不是 LCFS。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

from ..config import GeometryConfig
from .magnetic import MagneticFieldProvider


Direction = Literal["forward", "backward", "both"]


@dataclass(frozen=True)
class FieldLineBounds:
    """轴对称计算域及可选的 R-Z 第一壁多边形（单位 m）。"""

    R_bounds_m: tuple[float, float]
    Z_bounds_m: tuple[float, float]
    wall_polygon_RZ_m: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        r0, r1 = (float(self.R_bounds_m[0]), float(self.R_bounds_m[1]))
        z0, z1 = (float(self.Z_bounds_m[0]), float(self.Z_bounds_m[1]))
        if not all(math.isfinite(value) for value in (r0, r1, z0, z1)):
            raise ValueError("场线边界必须为有限数")
        if r0 < 0.0:
            raise ValueError("柱坐标 R 的下界不能为负")
        if r1 <= r0 or z1 <= z0:
            raise ValueError("场线边界必须满足上界 > 下界")
        object.__setattr__(self, "R_bounds_m", (r0, r1))
        object.__setattr__(self, "Z_bounds_m", (z0, z1))
        if self.wall_polygon_RZ_m is not None:
            polygon = tuple((float(R), float(Z)) for R, Z in self.wall_polygon_RZ_m)
            if len(polygon) >= 2 and polygon[0] == polygon[-1]:
                polygon = polygon[:-1]
            if len(polygon) < 3:
                raise ValueError("第一壁多边形至少需要 3 个顶点")
            if any(not (math.isfinite(R) and math.isfinite(Z)) or R < 0.0 for R, Z in polygon):
                raise ValueError("第一壁多边形坐标必须为有限数且 R >= 0")
            area2 = sum(
                polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
                - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
                for index in range(len(polygon))
            )
            if abs(area2) <= 1.0e-12:
                raise ValueError("第一壁多边形面积不能为零")
            object.__setattr__(self, "wall_polygon_RZ_m", polygon)

    @classmethod
    def from_geometry(cls, geometry: GeometryConfig) -> "FieldLineBounds":
        polygon = getattr(geometry, "wall_polygon_RZ_m", None)
        return cls(
            tuple(geometry.R_bounds_m),
            tuple(geometry.Z_bounds_m),
            None if polygon is None else tuple(tuple(point) for point in polygon),
        )

    @classmethod
    def coerce(cls, value: Any) -> "FieldLineBounds":
        """Convert a bounds object, ``GeometryConfig`` or pair of pairs."""

        if isinstance(value, cls):
            return value
        if isinstance(value, GeometryConfig):
            return cls.from_geometry(value)
        if isinstance(value, Mapping):
            # Accept both concise and schema-like spellings for lightweight
            # use from notebooks/JSON readers.
            r_bounds = value.get("R_bounds_m", value.get("r_bounds_m"))
            z_bounds = value.get("Z_bounds_m", value.get("z_bounds_m"))
            if r_bounds is not None and z_bounds is not None:
                polygon = value.get("wall_polygon_RZ_m", value.get("wall_polygon_rz_m"))
                return cls(
                    tuple(r_bounds),
                    tuple(z_bounds),
                    None if polygon is None else tuple(tuple(point) for point in polygon),
                )
        try:
            r_bounds, z_bounds = value
            return cls(tuple(r_bounds), tuple(z_bounds))
        except (TypeError, ValueError, IndexError) as exc:
            raise TypeError(
                "bounds 必须是 FieldLineBounds、GeometryConfig 或 ((Rmin,Rmax),(Zmin,Zmax))"
            ) from exc

    def contains(self, R_m: float, Z_m: float, tolerance_m: float = 0.0) -> bool:
        """Return whether a point lies inside (or on) the wall."""

        tol = max(0.0, float(tolerance_m))
        in_rectangle = (
            self.R_bounds_m[0] - tol <= float(R_m) <= self.R_bounds_m[1] + tol
            and self.Z_bounds_m[0] - tol <= float(Z_m) <= self.Z_bounds_m[1] + tol
        )
        if not in_rectangle or self.wall_polygon_RZ_m is None:
            return in_rectangle
        R = float(R_m)
        Z = float(Z_m)
        inside = False
        for index, start in enumerate(self.wall_polygon_RZ_m):
            end = self.wall_polygon_RZ_m[(index + 1) % len(self.wall_polygon_RZ_m)]
            distance = self._point_segment_distance(R, Z, start, end)
            if distance <= tol:
                return True
            R0, Z0 = start
            R1, Z1 = end
            if (Z0 > Z) != (Z1 > Z) and R < (R1 - R0) * (Z - Z0) / (Z1 - Z0) + R0:
                inside = not inside
        return inside

    @staticmethod
    def _point_segment_distance(
        R_m: float,
        Z_m: float,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        dR = end[0] - start[0]
        dZ = end[1] - start[1]
        length2 = dR * dR + dZ * dZ
        if length2 <= 0.0:
            return math.hypot(R_m - start[0], Z_m - start[1])
        fraction = max(
            0.0,
            min(1.0, ((R_m - start[0]) * dR + (Z_m - start[1]) * dZ) / length2),
        )
        return math.hypot(
            R_m - (start[0] + fraction * dR),
            Z_m - (start[1] + fraction * dZ),
        )

    def boundary_at(self, R_m: float, Z_m: float, tolerance_m: float = 1.0e-12) -> str | None:
        """Return the nearest wall side when a point is on/outside it."""

        R = float(R_m)
        Z = float(Z_m)
        tol = max(0.0, float(tolerance_m))
        if self.wall_polygon_RZ_m is not None:
            distances = [
                self._point_segment_distance(
                    R,
                    Z,
                    start,
                    self.wall_polygon_RZ_m[(index + 1) % len(self.wall_polygon_RZ_m)],
                )
                for index, start in enumerate(self.wall_polygon_RZ_m)
            ]
            nearest = min(range(len(distances)), key=distances.__getitem__)
            if distances[nearest] <= tol or not self.contains(R, Z, tolerance_m=tol):
                return f"wall_segment_{nearest}"
            return None
        candidates: list[tuple[float, str]] = []
        r_min, r_max = self.R_bounds_m
        z_min, z_max = self.Z_bounds_m
        if R <= r_min + tol:
            candidates.append((abs(R - r_min), "R_min"))
        if R >= r_max - tol:
            candidates.append((abs(R - r_max), "R_max"))
        if Z <= z_min + tol:
            candidates.append((abs(Z - z_min), "Z_min"))
        if Z >= z_max - tol:
            candidates.append((abs(Z - z_max), "Z_max"))
        return min(candidates)[1] if candidates else None


@dataclass(frozen=True)
class FieldLineBranch:
    """沿一个方向积分得到的场线分支。"""

    direction: int
    status: str
    length_m: float
    end_R_m: float
    end_Z_m: float
    end_phi_rad: float
    hit_boundary: bool
    boundary: str | None
    closed: bool
    timed_out: bool
    min_B_T: float
    max_B_T: float
    mirror_ratio: float
    n_steps: int
    path_s_m: tuple[float, ...]
    path_R_m: tuple[float, ...]
    path_Z_m: tuple[float, ...]
    path_phi_rad: tuple[float, ...]

    @property
    def connection_length_m(self) -> float:
        return self.length_m

    @property
    def wall_hit(self) -> bool:
        """Compatibility alias for callers using the shorter terminology."""

        return self.hit_boundary

    @property
    def termination(self) -> str:
        return self.status

    def to_dict(self, include_path: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "direction": self.direction,
            "status": self.status,
            "length_m": self.length_m,
            "connection_length_m": self.connection_length_m,
            "end_R_m": self.end_R_m,
            "end_Z_m": self.end_Z_m,
            "end_phi_rad": self.end_phi_rad,
            "hit_boundary": self.hit_boundary,
            "boundary": self.boundary,
            "closed": self.closed,
            "timed_out": self.timed_out,
            "min_B_T": self.min_B_T,
            "max_B_T": self.max_B_T,
            "mirror_ratio": self.mirror_ratio,
            "n_steps": self.n_steps,
        }
        if include_path:
            result.update(
                {
                    "path_s_m": list(self.path_s_m),
                    "path_R_m": list(self.path_R_m),
                    "path_Z_m": list(self.path_Z_m),
                    "path_phi_rad": list(self.path_phi_rad),
                }
            )
        return result


@dataclass(frozen=True)
class FieldLineTrace:
    """结构化的单向或双向场线追踪结果。

    对 ``direction="both"``，``length_m`` 是两个分支长度之和；顶层终点
    取 forward 分支（若存在），而完整的两个终点始终可从 ``branches`` 或
    ``endpoints`` 读取。``closed`` 在双向模式下要求两个分支都闭合。
    """

    provider_configuration: str
    direction: str
    start_R_m: float
    start_Z_m: float
    start_phi_rad: float
    Ip_A: float
    t_s: float
    status: str
    length_m: float
    end_R_m: float
    end_Z_m: float
    end_phi_rad: float
    hit_boundary: bool
    boundary: str | None
    closed: bool
    timed_out: bool
    min_B_T: float
    max_B_T: float
    mirror_ratio: float
    branches: tuple[FieldLineBranch, ...]

    @property
    def connection_length_m(self) -> float:
        return self.length_m

    @property
    def wall_hit(self) -> bool:
        return self.hit_boundary

    @property
    def termination(self) -> str:
        return self.status

    @property
    def forward(self) -> FieldLineBranch | None:
        return next((branch for branch in self.branches if branch.direction > 0), None)

    @property
    def backward(self) -> FieldLineBranch | None:
        return next((branch for branch in self.branches if branch.direction < 0), None)

    @property
    def endpoints(self) -> tuple[tuple[float, float, float], ...]:
        return tuple((branch.end_R_m, branch.end_Z_m, branch.end_phi_rad) for branch in self.branches)

    @property
    def boundaries(self) -> tuple[str, ...]:
        return tuple(branch.boundary for branch in self.branches if branch.boundary is not None)

    def to_dict(self, include_path: bool = True) -> dict[str, Any]:
        return {
            "provider_configuration": self.provider_configuration,
            "direction": self.direction,
            "start_R_m": self.start_R_m,
            "start_Z_m": self.start_Z_m,
            "start_phi_rad": self.start_phi_rad,
            "Ip_A": self.Ip_A,
            "t_s": self.t_s,
            "status": self.status,
            "length_m": self.length_m,
            "connection_length_m": self.connection_length_m,
            "end_R_m": self.end_R_m,
            "end_Z_m": self.end_Z_m,
            "end_phi_rad": self.end_phi_rad,
            "hit_boundary": self.hit_boundary,
            "wall_hit": self.wall_hit,
            "boundary": self.boundary,
            "boundaries": list(self.boundaries),
            "closed": self.closed,
            "timed_out": self.timed_out,
            "min_B_T": self.min_B_T,
            "max_B_T": self.max_B_T,
            "mirror_ratio": self.mirror_ratio,
            "branches": [branch.to_dict(include_path=include_path) for branch in self.branches],
        }


@dataclass(frozen=True)
class _State:
    R_m: float
    Z_m: float
    phi_rad: float


class FieldLineTracer:
    """用固定弧长 RK4 步长追踪轴对称磁场线。

    Parameters
    ----------
    field:
        实现 ``MagneticFieldProvider`` 的对象；``CompositeMagneticField`` 可
        直接传入。
    bounds:
        ``FieldLineBounds``、``GeometryConfig`` 或两个边界元组。若省略，
        尝试从 ``field.geometry`` 推断。
    step_m:
        每个积分步的弧长，必须为正。
    max_length_m:
        单个方向的最大追踪长度；达到后以 ``timeout`` 结束。
    closure_tolerance_m:
        极向回返闭合容差。缺省为 ``max(1.5*step_m, 1e-5 m)``。
    min_B_T:
        磁场幅值下限；低于此值时以 ``zero_field`` 结束，避免除零。
    """

    def __init__(
        self,
        field: MagneticFieldProvider,
        bounds: FieldLineBounds | GeometryConfig | Sequence[Sequence[float]] | Mapping[str, Any] | None = None,
        *,
        geometry: GeometryConfig | None = None,
        R_bounds_m: Sequence[float] | None = None,
        Z_bounds_m: Sequence[float] | None = None,
        step_m: float = 1.0e-3,
        max_length_m: float = 100.0,
        closure_tolerance_m: float | None = None,
        min_B_T: float = 1.0e-10,
        max_steps: int | None = None,
    ) -> None:
        self.field = field
        if geometry is not None and bounds is not None:
            raise ValueError("geometry 与 bounds 不能同时提供")
        if geometry is not None and (R_bounds_m is not None or Z_bounds_m is not None):
            raise ValueError("geometry 与 R_bounds_m/Z_bounds_m 不能同时提供")
        if geometry is not None:
            bounds = geometry
        if bounds is not None and (R_bounds_m is not None or Z_bounds_m is not None):
            raise ValueError("bounds 与 R_bounds_m/Z_bounds_m 不能同时提供")
        if bounds is None and (R_bounds_m is not None or Z_bounds_m is not None):
            if R_bounds_m is None or Z_bounds_m is None:
                raise ValueError("R_bounds_m 与 Z_bounds_m 必须同时提供")
            bounds = (R_bounds_m, Z_bounds_m)
        if bounds is None:
            geometry = getattr(field, "geometry", None)
            if geometry is None:
                raise ValueError("无法从 field 推断边界；请显式提供 bounds")
            bounds = geometry
        self.bounds = FieldLineBounds.coerce(bounds)
        self.step_m = self._positive_finite(step_m, "step_m")
        self.max_length_m = self._positive_finite(max_length_m, "max_length_m")
        if closure_tolerance_m is None:
            closure_tolerance_m = max(1.5 * self.step_m, 1.0e-5)
        self.closure_tolerance_m = self._nonnegative_finite(closure_tolerance_m, "closure_tolerance_m")
        self.min_B_T = self._nonnegative_finite(min_B_T, "min_B_T")
        if max_steps is not None:
            if isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps < 1:
                raise ValueError("max_steps 必须是正整数")
            self.max_steps = int(max_steps)
        else:
            self.max_steps = None

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} 必须为有限正数")
        return result

    @staticmethod
    def _nonnegative_finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} 必须为有限非负数")
        return result

    @staticmethod
    def _directions(direction: str | int) -> tuple[str, tuple[int, ...]]:
        if isinstance(direction, int) and not isinstance(direction, bool):
            if direction == 1:
                return "forward", (1,)
            if direction == -1:
                return "backward", (-1,)
        normalized = str(direction).strip().lower()
        aliases = {
            "+": ("forward", (1,)),
            "forward": ("forward", (1,)),
            "fwd": ("forward", (1,)),
            "-": ("backward", (-1,)),
            "backward": ("backward", (-1,)),
            "back": ("backward", (-1,)),
            "reverse": ("backward", (-1,)),
            "negative": ("backward", (-1,)),
            "positive": ("forward", (1,)),
            "both": ("both", (1, -1)),
            "bidirectional": ("both", (1, -1)),
            "bi": ("both", (1, -1)),
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError("direction 必须是 forward、backward 或 both") from exc

    def _field_rhs(
        self,
        state: _State,
        Ip_A: float,
        t_s: float,
        sign: int,
        min_B_T: float,
    ) -> tuple[float, float, float, float] | None:
        point = self.field.evaluate(state.R_m, state.Z_m, Ip_A, t_s)
        try:
            b_r = float(point.B_R_T)
            b_z = float(point.B_Z_T)
            b_phi = float(point.B_phi_T)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("磁场 provider.evaluate 必须返回 B_R_T/B_Z_T/B_phi_T") from exc
        if not all(math.isfinite(value) for value in (b_r, b_z, b_phi)):
            return None
        magnitude = math.sqrt(b_r * b_r + b_z * b_z + b_phi * b_phi)
        if not math.isfinite(magnitude) or magnitude <= min_B_T:
            return None
        # R=0 is outside the usual tokamak domain.  Keep a finite denominator
        # for a custom wall that touches the axis, while terminating only when
        # the coordinate is truly singular and the toroidal field is non-zero.
        radius = float(state.R_m)
        if radius <= 0.0 and abs(b_phi) > min_B_T:
            return None
        radius = max(radius, 1.0e-12)
        factor = float(sign) / magnitude
        return factor * b_r, factor * b_z, factor * b_phi / radius, magnitude

    def _rk4_step(
        self,
        state: _State,
        ds_m: float,
        Ip_A: float,
        t_s: float,
        sign: int,
        min_B_T: float,
    ) -> tuple[_State | None, float | None]:
        """Advance one arc-length step; return ``None`` for a zero field."""

        first = self._field_rhs(state, Ip_A, t_s, sign, min_B_T)
        if first is None:
            return None, None

        def add(base: _State, deriv: tuple[float, float, float, float], scale: float) -> _State:
            return _State(
                base.R_m + scale * deriv[0],
                base.Z_m + scale * deriv[1],
                base.phi_rad + scale * deriv[2],
            )

        k1 = first
        k2_state = add(state, k1, 0.5 * ds_m)
        k2 = self._field_rhs(k2_state, Ip_A, t_s, sign, min_B_T)
        if k2 is None:
            return None, None
        k3_state = add(state, k2, 0.5 * ds_m)
        k3 = self._field_rhs(k3_state, Ip_A, t_s, sign, min_B_T)
        if k3 is None:
            return None, None
        k4_state = add(state, k3, ds_m)
        k4 = self._field_rhs(k4_state, Ip_A, t_s, sign, min_B_T)
        if k4 is None:
            return None, None
        next_state = _State(
            state.R_m + ds_m * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state.Z_m + ds_m * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state.phi_rad + ds_m * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        )
        return next_state, k4[3]

    def _boundary_crossing(
        self,
        old: _State,
        new: _State,
        bounds: FieldLineBounds,
    ) -> tuple[float, _State, str | None]:
        """Locate the first wall crossing by bisection along a step."""

        if bounds.contains(new.R_m, new.Z_m):
            return 1.0, new, None
        low = 0.0
        high = 1.0
        for _ in range(60):
            middle = 0.5 * (low + high)
            R = old.R_m + middle * (new.R_m - old.R_m)
            Z = old.Z_m + middle * (new.Z_m - old.Z_m)
            if bounds.contains(R, Z):
                low = middle
            else:
                high = middle
        alpha = low
        endpoint = _State(
            old.R_m + alpha * (new.R_m - old.R_m),
            old.Z_m + alpha * (new.Z_m - old.Z_m),
            old.phi_rad + alpha * (new.phi_rad - old.phi_rad),
        )
        # Use the extrapolated point to identify the side, then fall back to
        # the clipped endpoint for corner/tolerance cases.
        boundary = bounds.boundary_at(new.R_m, new.Z_m)
        if boundary is None:
            boundary = bounds.boundary_at(endpoint.R_m, endpoint.Z_m, tolerance_m=1.0e-8)
        return alpha, endpoint, boundary

    def _trace_branch(
        self,
        start: _State,
        Ip_A: float,
        t_s: float,
        sign: int,
        step_m: float,
        max_length_m: float,
        closure_tolerance_m: float,
        max_steps: int | None,
        bounds: FieldLineBounds,
        min_B_T: float,
    ) -> FieldLineBranch:
        path_s = [0.0]
        path_R = [start.R_m]
        path_Z = [start.Z_m]
        path_phi = [start.phi_rad]
        if not bounds.contains(start.R_m, start.Z_m):
            status = "outside_boundary"
            return FieldLineBranch(
                sign,
                status,
                0.0,
                start.R_m,
                start.Z_m,
                start.phi_rad,
                False,
                bounds.boundary_at(start.R_m, start.Z_m),
                False,
                False,
                0.0,
                0.0,
                0.0,
                0,
                tuple(path_s),
                tuple(path_R),
                tuple(path_Z),
                tuple(path_phi),
            )
        initial_rhs = self._field_rhs(start, Ip_A, t_s, sign, min_B_T)
        if initial_rhs is None:
            status = "zero_field"
            return FieldLineBranch(
                sign,
                status,
                0.0,
                start.R_m,
                start.Z_m,
                start.phi_rad,
                False,
                None,
                False,
                False,
                0.0,
                0.0,
                math.inf if status == "zero_field" else 0.0,
                0,
                tuple(path_s),
                tuple(path_R),
                tuple(path_Z),
                tuple(path_phi),
            )

        min_B = initial_rhs[3]
        max_B = initial_rhs[3]
        state = start
        length = 0.0
        n_steps = 0
        status = "timeout"
        boundary: str | None = None
        closed = False
        timed_out = False
        # Do not classify nearby numerical points as a return.  A branch must
        # first leave the start neighbourhood and only then re-enter it.
        min_closure_length = max(2.0 * step_m, 2.0 * closure_tolerance_m)
        left_start_neighbourhood = False
        # A purely toroidal field has no poloidal displacement.  It is still a
        # closed axisymmetric orbit after one toroidal circumference.
        toroidal_loop_length = 2.0 * math.pi * max(abs(start.R_m), 1.0e-12)

        while length < max_length_m - 1.0e-15:
            if max_steps is not None and n_steps >= max_steps:
                timed_out = True
                status = "timeout"
                break
            ds = min(step_m, max_length_m - length)
            next_state, endpoint_B = self._rk4_step(state, ds, Ip_A, t_s, sign, min_B_T)
            if next_state is None:
                status = "zero_field"
                break
            alpha, endpoint, crossing = self._boundary_crossing(state, next_state, bounds)
            actual_ds = ds * alpha
            length += actual_ds
            n_steps += 1
            state = endpoint
            path_s.append(length)
            path_R.append(state.R_m)
            path_Z.append(state.Z_m)
            path_phi.append(state.phi_rad)
            endpoint_point = self.field.evaluate(state.R_m, state.Z_m, Ip_A, t_s)
            endpoint_B_value = math.sqrt(
                float(endpoint_point.B_R_T) ** 2
                + float(endpoint_point.B_Z_T) ** 2
                + float(endpoint_point.B_phi_T) ** 2
            )
            if math.isfinite(endpoint_B_value):
                min_B = min(min_B, endpoint_B_value)
                max_B = max(max_B, endpoint_B_value)

            if crossing is not None or not bounds.contains(state.R_m, state.Z_m):
                boundary = crossing or bounds.boundary_at(state.R_m, state.Z_m)
                status = "wall_hit"
                break
            distance_from_start = math.hypot(state.R_m - start.R_m, state.Z_m - start.Z_m)
            if distance_from_start > closure_tolerance_m:
                left_start_neighbourhood = True
            if (
                not left_start_neighbourhood
                and length >= max(min_closure_length, toroidal_loop_length)
                and distance_from_start <= closure_tolerance_m
            ):
                closed = True
                status = "closed"
                break
            if (
                left_start_neighbourhood
                and length >= min_closure_length
                and distance_from_start <= closure_tolerance_m
            ):
                closed = True
                status = "closed"
                break
            if length >= max_length_m - 1.0e-15:
                timed_out = True
                status = "timeout"
                break

        if min_B <= min_B_T:
            mirror_ratio = math.inf
        else:
            mirror_ratio = max_B / min_B
        return FieldLineBranch(
            sign,
            status,
            length,
            state.R_m,
            state.Z_m,
            state.phi_rad,
            status == "wall_hit",
            boundary,
            closed,
            timed_out or status == "timeout",
            min_B,
            max_B,
            mirror_ratio,
            n_steps,
            tuple(path_s),
            tuple(path_R),
            tuple(path_Z),
            tuple(path_phi),
        )

    def trace(
        self,
        start_R_m: float | None = None,
        start_Z_m: float | None = None,
        *,
        start: Sequence[float] | None = None,
        R_m: float | None = None,
        Z_m: float | None = None,
        start_phi_rad: float = 0.0,
        phi_rad: float | None = None,
        Ip_A: float = 0.0,
        t_s: float = 0.0,
        direction: str | int = "both",
        step_m: float | None = None,
        max_length_m: float | None = None,
        closure_tolerance_m: float | None = None,
        min_B_T: float | None = None,
        max_steps: int | None = None,
        bounds: FieldLineBounds | GeometryConfig | Sequence[Sequence[float]] | Mapping[str, Any] | None = None,
    ) -> FieldLineTrace:
        """Trace one or both directions from an interior point.

        ``R_m``/``Z_m`` and ``phi_rad`` are accepted as aliases for the
        ``start_*`` names to make interactive use concise.
        """

        if start is not None:
            if len(start) != 2:
                raise ValueError("start 必须是 (R_m, Z_m)")
            if start_R_m is not None and float(start_R_m) != float(start[0]):
                raise ValueError("start_R_m 与 start[0] 不一致")
            if start_Z_m is not None and float(start_Z_m) != float(start[1]):
                raise ValueError("start_Z_m 与 start[1] 不一致")
            start_R_m, start_Z_m = float(start[0]), float(start[1])
        if start_R_m is None:
            start_R_m = R_m
        elif R_m is not None and float(R_m) != float(start_R_m):
            raise ValueError("start_R_m 与 R_m 不一致")
        if start_Z_m is None:
            start_Z_m = Z_m
        elif Z_m is not None and float(Z_m) != float(start_Z_m):
            raise ValueError("start_Z_m 与 Z_m 不一致")
        if start_R_m is None or start_Z_m is None:
            raise TypeError("trace 需要 start_R_m 和 start_Z_m")
        if phi_rad is not None:
            if start_phi_rad != 0.0 and not math.isclose(start_phi_rad, phi_rad, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("start_phi_rad 与 phi_rad 不一致")
            start_phi_rad = float(phi_rad)
        R0 = float(start_R_m)
        Z0 = float(start_Z_m)
        phi0 = float(start_phi_rad)
        ip = float(Ip_A)
        time = float(t_s)
        if not all(math.isfinite(value) for value in (R0, Z0, phi0, ip, time)):
            raise ValueError("起点、Ip_A 和 t_s 必须为有限数")
        local_bounds = self.bounds if bounds is None else FieldLineBounds.coerce(bounds)
        local_step = self.step_m if step_m is None else self._positive_finite(step_m, "step_m")
        local_max_length = self.max_length_m if max_length_m is None else self._positive_finite(max_length_m, "max_length_m")
        local_tolerance = (
            self.closure_tolerance_m
            if closure_tolerance_m is None
            else self._nonnegative_finite(closure_tolerance_m, "closure_tolerance_m")
        )
        local_min_B = self.min_B_T if min_B_T is None else self._nonnegative_finite(min_B_T, "min_B_T")
        local_max_steps = self.max_steps if max_steps is None else max_steps
        if local_max_steps is not None:
            if isinstance(local_max_steps, bool) or int(local_max_steps) != local_max_steps or local_max_steps < 1:
                raise ValueError("max_steps 必须是正整数")
            local_max_steps = int(local_max_steps)

        # Keep per-call bounds/floor isolated from the tracer defaults while
        # reusing the same implementation object.
        normalized_direction, signs = self._directions(direction)
        start = _State(R0, Z0, phi0)
        branches = tuple(
            self._trace_branch(
                start,
                ip,
                time,
                sign,
                local_step,
                local_max_length,
                local_tolerance,
                local_max_steps,
                local_bounds,
                local_min_B,
            )
            for sign in signs
        )

        # Aggregate branch diagnostics.  ``branches`` is never empty because
        # _directions validates the requested mode.
        all_closed = all(branch.closed for branch in branches)
        any_wall = any(branch.hit_boundary for branch in branches)
        any_timeout = any(branch.timed_out for branch in branches)
        if all_closed:
            status = "closed"
        elif any_wall:
            status = "wall_hit"
        elif any(branch.status == "outside_boundary" for branch in branches):
            status = "outside_boundary"
        elif any(branch.status == "zero_field" for branch in branches):
            status = "zero_field"
        elif any_timeout:
            status = "timeout"
        else:
            status = branches[0].status
        primary = next((branch for branch in branches if branch.direction > 0), branches[0])
        finite_min = [branch.min_B_T for branch in branches if branch.max_B_T > 0.0]
        min_B = min(finite_min) if finite_min else 0.0
        max_B = max((branch.max_B_T for branch in branches), default=0.0)
        mirror_ratio = math.inf if min_B <= 0.0 else max_B / min_B
        return FieldLineTrace(
            provider_configuration=str(getattr(self.field, "configuration", "custom")),
            direction=normalized_direction,
            start_R_m=R0,
            start_Z_m=Z0,
            start_phi_rad=phi0,
            Ip_A=ip,
            t_s=time,
            status=status,
            length_m=sum(branch.length_m for branch in branches),
            end_R_m=primary.end_R_m,
            end_Z_m=primary.end_Z_m,
            end_phi_rad=primary.end_phi_rad,
            hit_boundary=any_wall,
            boundary=primary.boundary if len(branches) == 1 else (primary.boundary if len(set(filter(None, (branch.boundary for branch in branches)))) == 1 else None),
            closed=all_closed,
            timed_out=any_timeout,
            min_B_T=min_B,
            max_B_T=max_B,
            mirror_ratio=mirror_ratio,
            branches=branches,
        )


# A descriptive alias for code that prefers the geometry-specific name.
AxisymmetricFieldLineTracer = FieldLineTracer


def trace_field_line(
    field: MagneticFieldProvider,
    start_R_m: float,
    start_Z_m: float,
    **kwargs: Any,
) -> FieldLineTrace:
    """Functional convenience wrapper around :class:`FieldLineTracer`."""

    tracer = FieldLineTracer(field, **{key: kwargs.pop(key) for key in ("bounds", "R_bounds_m", "Z_bounds_m", "step_m", "max_length_m", "closure_tolerance_m", "min_B_T", "max_steps") if key in kwargs})
    return tracer.trace(start_R_m, start_Z_m, **kwargs)


__all__ = [
    "AxisymmetricFieldLineTracer",
    "FieldLineBounds",
    "FieldLineBranch",
    "FieldLineTrace",
    "FieldLineTracer",
    "trace_field_line",
]
