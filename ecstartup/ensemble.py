"""参数不确定度/启动概率的轻量 ensemble 工作流。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .comparison import summarize_device
from .config import SimulationConfig
from .solver import integrate


@dataclass(frozen=True)
class EnsembleResult:
    records: list[dict[str, Any]]
    success_event: str
    success_probability: float
    wilson_95_low: float
    wilson_95_high: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_event": self.success_event,
            "success_probability": self.success_probability,
            "wilson_95": [self.wilson_95_low, self.wilson_95_high],
            "n_samples": len(self.records),
            "records": self.records,
        }


def _set_dotted(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = mapping
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _sample_value(rng: np.random.Generator, spec: Any) -> float:
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        low, high = map(float, spec)
        logarithmic = low > 0.0
    elif isinstance(spec, Mapping):
        low = float(spec["low"])
        high = float(spec["high"])
        logarithmic = bool(spec.get("log", low > 0.0))
    else:
        raise ValueError("参数范围必须是 [low, high] 或 {low, high, log}")
    if high < low or (logarithmic and low <= 0.0):
        raise ValueError("参数范围无效")
    if logarithmic:
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    return float(rng.uniform(low, high))


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return max(0.0, center - half), min(1.0, center + half)


def run_ensemble(
    base_config: SimulationConfig,
    parameter_ranges: Mapping[str, Any],
    samples: int = 16,
    seed: int | None = None,
    success_event: str = "preionization",
) -> EnsembleResult:
    """在给定先验范围内采样并估计启动事件概率。

    `parameter_ranges` 使用点号路径，例如 `gas.pressure_Pa` 或
    `geometry.B_vertical_T`。该函数不声称给出统计学上的装置可靠性；它只
    把截面、壁参数和机器参数不确定度传播到模型事件。
    """

    if samples < 1:
        raise ValueError("samples 必须为正")
    rng = np.random.default_rng(base_config.seed if seed is None else seed)
    records: list[dict[str, Any]] = []
    successes = 0
    for index in range(samples):
        raw = base_config.model_dump(mode="python")
        sampled: dict[str, float] = {}
        for path, spec in parameter_ranges.items():
            value = _sample_value(rng, spec)
            _set_dotted(raw, path, value)
            sampled[path] = value
        config = SimulationConfig.model_validate(raw)
        result = integrate(config)
        summary = summarize_device(f"sample_{index:04d}", config, result)
        event_types = {event.type for event in result.events}
        success = success_event in event_types
        successes += int(success)
        records.append({"index": index, "success": success, "sampled": sampled, **summary.as_dict()})
    probability = successes / samples
    low, high = _wilson(successes, samples)
    return EnsembleResult(records, success_event, probability, low, high)
