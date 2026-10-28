"""阶段事件、启动判据和诊断记录。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

from .config import ThresholdConfig
from .state import Event, StartupStage


class StageManager:
    """把连续物理量映射为可审计的阶段状态机。"""

    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.thresholds = thresholds
        self.stage = StartupStage.VACUUM
        self._breakdown_count = 0
        self._preionization_count = 0
        self._cfs_count = 0

    def update(self, time_s: float, values: Dict[str, float]) -> Optional[Event]:
        ne = values.get("ne_m3", 0.0)
        te = values.get("Te_eV", 0.0)
        growth = values.get("net_growth_s", -1.0)
        closed = values.get("closed_fraction", 0.0)
        persistence = self.thresholds.persistence_steps

        if self.stage not in {StartupStage.CFS_ESTABLISHED, StartupStage.FAILED} and ne < self.thresholds.failure_density_m3:
            old = self.stage
            self.stage = StartupStage.FAILED
            return Event("startup_failed", time_s, old.value, self.stage.value, {"ne_m3": ne})

        if self.stage == StartupStage.VACUUM:
            self._breakdown_count = self._breakdown_count + 1 if (
                ne >= self.thresholds.breakdown_density_m3 and growth > 0.0
            ) else 0
            if self._breakdown_count >= persistence:
                old = self.stage
                self.stage = StartupStage.BREAKDOWN
                return Event(
                    "breakdown",
                    time_s,
                    old.value,
                    self.stage.value,
                    {"ne_m3": ne, "net_growth_s": growth},
                )

        elif self.stage == StartupStage.BREAKDOWN:
            self._preionization_count = self._preionization_count + 1 if (
                ne >= self.thresholds.preionization_density_m3
                and te >= self.thresholds.preionization_Te_eV
                and growth > 0.0
            ) else 0
            if self._preionization_count >= persistence:
                old = self.stage
                self.stage = StartupStage.PRE_IONIZATION
                return Event(
                    "preionization",
                    time_s,
                    old.value,
                    self.stage.value,
                    {"ne_m3": ne, "Te_eV": te, "net_growth_s": growth},
                )

        elif self.stage == StartupStage.PRE_IONIZATION:
            if closed > 0.0:
                old = self.stage
                self.stage = StartupStage.CFS_FORMING
                self._cfs_count = 1
                return Event(
                    "cfs_candidate",
                    time_s,
                    old.value,
                    self.stage.value,
                    {"closed_fraction": closed},
                )

        elif self.stage == StartupStage.CFS_FORMING:
            self._cfs_count = self._cfs_count + 1 if closed >= self.thresholds.cfs_closed_fraction else 0
            if self._cfs_count >= persistence:
                old = self.stage
                self.stage = StartupStage.CFS_ESTABLISHED
                return Event(
                    "cfs",
                    time_s,
                    old.value,
                    self.stage.value,
                    {"closed_fraction": closed, "persistence_steps": self._cfs_count},
                )
        return None


def add_stage(values: Dict[str, float], stage: StartupStage) -> Dict[str, float | str]:
    result: Dict[str, float | str] = dict(values)
    result["stage"] = stage.value
    return result


def event_to_dict(event: Event) -> dict:
    return asdict(event)
