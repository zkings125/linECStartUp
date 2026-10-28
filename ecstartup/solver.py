"""刚性 ODE 时间推进和结果对象。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp

from .config import SimulationConfig
from .diagnostics import StageManager, add_stage
from .physics.global_model import GlobalStartupModel
from .state import Event, GlobalState


@dataclass
class RunResult:
    times_s: np.ndarray
    states: List[Any]
    diagnostics: List[Dict[str, float | str]]
    events: List[Event]
    success: bool
    message: str
    kinetic_history: Any | None = None

    @property
    def final_state(self) -> Any:
        return self.states[-1]


def _time_grid(config: SimulationConfig) -> np.ndarray:
    cadence = config.outputs.cadence_s or config.numerics.dt_s
    start = config.numerics.t_start_s
    end = config.numerics.t_end_s
    n = max(1, int(np.ceil((end - start) / cadence)))
    times = np.linspace(start, end, n + 1)
    times[0] = start
    times[-1] = config.numerics.t_end_s
    return times


def _integrate_global(
    config: SimulationConfig, model: GlobalStartupModel | None = None
) -> RunResult:
    """运行一个 0D 案例并在输出网格上计算阶段诊断。

    ``model`` 可用于注入真实磁场、率模型或电流闭合；省略时保持默认
    的解析 surrogate 装配方式。
    """

    model = model if model is not None else GlobalStartupModel(config)
    initial = model.initial_state()
    t_eval = _time_grid(config)
    max_step = config.numerics.max_step_s or config.numerics.dt_s
    # 状态布局由模型拥有；启用多物种后向量长度不再固定为 7。
    tolerance_provider = getattr(model, "absolute_tolerance_vector", None)
    if callable(tolerance_provider):
        atol = np.asarray(tolerance_provider(config.numerics.atol), dtype=float)
    else:
        atol = np.asarray([1.0e5, 1.0e5, 1.0e10, 1.0e-8, 1.0e-3, 1.0e5, 1.0e-8])
        atol = np.maximum(atol, config.numerics.atol)
    solution = solve_ivp(
        model.rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        initial.to_vector(),
        method=config.numerics.solver,
        t_eval=t_eval,
        max_step=max_step,
        rtol=config.numerics.rtol,
        atol=atol,
    )
    state_parser = getattr(model, "state_from_vector", GlobalState.from_vector)
    states = [state_parser(solution.y[:, i]).projected() for i in range(solution.y.shape[1])]
    manager = StageManager(config.thresholds)
    diagnostics: List[Dict[str, float | str]] = []
    events: List[Event] = []
    for t_s, state in zip(solution.t, states):
        evaluation = model.evaluate(float(t_s), state)
        row = add_stage(evaluation.values, manager.stage)
        event = manager.update(float(t_s), evaluation.values)
        if event is not None:
            events.append(event)
        # 事件发生的这个采样点归入新阶段，便于画阶跃图。
        row["stage"] = manager.stage.value
        diagnostics.append(row)
    return RunResult(solution.t, states, diagnostics, events, bool(solution.success), solution.message)


def integrate(config: SimulationConfig, model: GlobalStartupModel | None = None) -> RunResult:
    """Run the configured global model, optionally with time-resolved F-P feedback."""

    if model is None and config.kinetic.coupling_mode == "time_dependent_feedback":
        if config.kinetic.backend == "open_field_line_1d2v":
            from .physics.spatial_time_dependent_kinetic import (
                integrate_bidirectional_spatial_kinetic,
            )

            return integrate_bidirectional_spatial_kinetic(config, _integrate_global)
        from .physics.time_dependent_kinetic import integrate_bidirectional_kinetic

        return integrate_bidirectional_kinetic(config, _integrate_global)
    return _integrate_global(config, model)
