"""Experimental-shot ingestion and conservative validation labels.

The EXL-50U CSV exports store every diagnostic as an independent ``time`` /
``data`` column pair.  This module preserves those native time bases and keeps
raw-to-SI calibration provenance explicit.  A channel is never silently
treated as SI when its calibration is unknown.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SignalCalibration:
    canonical_name: str
    raw_unit: str
    physical_unit: str | None
    scale_to_physical: float | None
    offset_to_physical: float = 0.0
    status: str = "unknown"
    note: str = ""

    @classmethod
    def from_mapping(cls, name: str, payload: Mapping[str, object]) -> "SignalCalibration":
        scale = payload.get("scale_to_physical")
        return cls(
            canonical_name=name,
            raw_unit=str(payload.get("raw_unit", "unknown")),
            physical_unit=(
                None if payload.get("physical_unit") is None else str(payload["physical_unit"])
            ),
            scale_to_physical=None if scale is None else float(scale),
            offset_to_physical=float(payload.get("offset_to_physical", 0.0)),
            status=str(payload.get("status", "unknown")),
            note=str(payload.get("note", "")),
        )

    def convert(self, values: np.ndarray) -> np.ndarray | None:
        if self.scale_to_physical is None:
            return None
        return self.offset_to_physical + self.scale_to_physical * np.asarray(values, dtype=float)

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_name": self.canonical_name,
            "raw_unit": self.raw_unit,
            "physical_unit": self.physical_unit,
            "scale_to_physical": self.scale_to_physical,
            "offset_to_physical": self.offset_to_physical,
            "status": self.status,
            "note": self.note,
        }


@dataclass(frozen=True)
class ShotSignal:
    name: str
    time_s: np.ndarray
    raw_values: np.ndarray
    calibration: SignalCalibration

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float)
        values = np.asarray(self.raw_values, dtype=float)
        if time.ndim != 1 or values.ndim != 1 or len(time) != len(values):
            raise ValueError("shot signal time/data arrays must be one-dimensional and equal length")
        if len(time) and (np.any(~np.isfinite(time)) or np.any(~np.isfinite(values))):
            raise ValueError("shot signal contains non-finite values")
        if len(time) > 1 and np.any(np.diff(time) < 0.0):
            order = np.argsort(time, kind="stable")
            time = time[order]
            values = values[order]
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "raw_values", values)

    @property
    def physical_values(self) -> np.ndarray | None:
        return self.calibration.convert(self.raw_values)

    def waveform(self, *, physical: bool = True) -> tuple[tuple[float, float], ...]:
        values = self.physical_values if physical else self.raw_values
        if values is None:
            raise ValueError(f"signal {self.name} has no physical calibration")
        return tuple((float(t), float(value)) for t, value in zip(self.time_s, values))

    def summary(self) -> dict[str, object]:
        raw_min = float(np.min(self.raw_values)) if len(self.raw_values) else None
        raw_max = float(np.max(self.raw_values)) if len(self.raw_values) else None
        physical = self.physical_values
        return {
            "sample_count": int(len(self.time_s)),
            "time_range_s": (
                [float(self.time_s[0]), float(self.time_s[-1])] if len(self.time_s) else None
            ),
            "raw_range": [raw_min, raw_max] if len(self.raw_values) else None,
            "physical_range": (
                [float(np.min(physical)), float(np.max(physical))]
                if physical is not None and len(physical)
                else None
            ),
            "calibration": self.calibration.to_dict(),
        }


@dataclass(frozen=True)
class EXL50UShot:
    shot: int
    source_path: Path
    signals: Mapping[str, ShotSignal]
    warnings: tuple[str, ...] = ()

    def require(self, name: str) -> ShotSignal:
        try:
            return self.signals[name]
        except KeyError as exc:
            raise KeyError(f"shot {self.shot} does not contain channel {name!r}") from exc

    def coil_waveforms_A(self) -> dict[str, tuple[tuple[float, float], ...]]:
        names = [f"I_PF{index}" for index in range(1, 11)] + ["I_CS"]
        return {
            name: self.require(name).waveform(physical=True)
            for name in names
            if name in self.signals
        }

    def validation_label(
        self,
        *,
        minimum_peak_current_A: float = 100.0e3,
        retained_fraction_for_success: float = 0.75,
        retained_fraction_for_termination: float = 0.60,
    ) -> dict[str, object]:
        """Return a waveform-derived candidate label, never an experiment log label."""

        if "ip" not in self.signals:
            return {
                "candidate": "unclassified",
                "reason": "Ip channel is missing",
                "label_authority": "waveform_heuristic_only",
            }
        signal = self.signals["ip"]
        current = signal.physical_values
        if current is None or len(current) < 3:
            return {
                "candidate": "unclassified",
                "reason": "Ip calibration or samples are missing",
                "label_authority": "waveform_heuristic_only",
            }
        peak_index = int(np.argmax(current))
        peak = float(current[peak_index])
        tail_count = max(1, int(math.ceil(0.05 * len(current))))
        final = float(np.median(current[-tail_count:]))
        retained = final / peak if peak > 0.0 else 0.0
        peak_time = float(signal.time_s[peak_index])
        candidate = "failed_or_inconclusive"
        reason = "peak plasma current did not pass the configured startup criterion"
        if peak >= minimum_peak_current_A and retained >= retained_fraction_for_success:
            candidate = "startup_success_candidate"
            reason = "Ip exceeded the threshold and remained high through the recorded window"
        elif peak >= minimum_peak_current_A and retained <= retained_fraction_for_termination:
            candidate = "termination_candidate"
            reason = "Ip exceeded the threshold and then fell before the end of the recorded window"

        h_alpha = self.signals.get("HA001")
        h_alpha_evidence: dict[str, object] | None = None
        if h_alpha is not None and len(h_alpha.raw_values):
            h_index = int(np.argmax(h_alpha.raw_values))
            h_time = float(h_alpha.time_s[h_index])
            baseline = float(np.median(h_alpha.raw_values))
            h_alpha_evidence = {
                "peak_raw": float(h_alpha.raw_values[h_index]),
                "peak_time_s": h_time,
                "peak_over_median": float(h_alpha.raw_values[h_index] / max(abs(baseline), 1.0e-30)),
                "time_from_Ip_peak_s": h_time - peak_time,
                "coincident_with_Ip_peak_3ms": abs(h_time - peak_time) <= 3.0e-3,
            }
        return {
            "candidate": candidate,
            "reason": reason,
            "label_authority": "waveform_heuristic_only; confirm against shot log",
            "peak_Ip_A": peak,
            "peak_Ip_time_s": peak_time,
            "final_Ip_A": final,
            "retained_Ip_fraction": retained,
            "H_alpha_evidence": h_alpha_evidence,
        }

    def summary(self) -> dict[str, object]:
        return {
            "shot": self.shot,
            "source_path": str(self.source_path),
            "time_base": "independent_per_channel",
            "startup_window_note": "CSV covers approximately -100 to +100 ms; no GS/EFIT surfaces are assumed",
            "validation_label": self.validation_label(),
            "signals": {name: signal.summary() for name, signal in self.signals.items()},
            "warnings": list(self.warnings),
        }


def _default_calibration(name: str) -> SignalCalibration:
    if name == "ip":
        return SignalCalibration(
            name, "kA", "A", 1.0e3, status="provisional",
            note="Scale inferred from EXL-50U waveform magnitude; confirm against archive metadata.",
        )
    if name.startswith("I_PF") or name == "I_CS":
        return SignalCalibration(
            name, "A", "A", 1.0, status="provisional",
            note="Channel name and magnitude are consistent with amperes; polarity is retained.",
        )
    if name == "LOOPV":
        return SignalCalibration(name, "V", "V", 1.0, status="provisional")
    if name == "BT":
        return SignalCalibration(name, "T", "T", 1.0, status="provisional")
    if name == "HCN_NE001":
        return SignalCalibration(
            name, "10^19 m^-3", "m^-3", 1.0e19, status="provisional",
            note="Density scale is conventional for this export and must be checked against diagnostic metadata.",
        )
    if name == "ITF01":
        return SignalCalibration(
            name, "archive unit", None, None, status="unknown",
            note="Use measured BT unless the TF-current calibration is supplied.",
        )
    if name == "GAS_PRES01":
        return SignalCalibration(
            name, "archive unit", None, None, status="unknown",
            note="Do not convert to Pa/mPa without the gauge calibration and location.",
        )
    if name.startswith("ECRH"):
        return SignalCalibration(
            name, "monitor voltage", None, None, status="unknown",
            note="A gyrotron-specific monitor-to-power calibration is required.",
        )
    if name.startswith("HA") or name.startswith("HXR") or name == "GAS_OUT01":
        return SignalCalibration(name, "a.u.", None, None, status="relative_only")
    return SignalCalibration(name, "unknown", None, None, status="unknown")


def _column_pairs(header: Sequence[str], shot: int) -> dict[str, tuple[int, int]]:
    suffix = f"{shot}time"
    data_suffix = f"{shot}data"
    positions = {name: index for index, name in enumerate(header)}
    pairs: dict[str, tuple[int, int]] = {}
    for time_name, time_index in positions.items():
        if not time_name.endswith(suffix):
            continue
        canonical = time_name[: -len(suffix)]
        data_name = f"{canonical}{data_suffix}"
        if data_name not in positions:
            raise ValueError(f"missing data column paired with {time_name!r}")
        pairs[canonical] = (time_index, positions[data_name])
    if not pairs:
        raise ValueError(f"no EXL-50U time/data column pairs found for shot {shot}")
    return pairs


def _extrema_downsample(
    time_s: np.ndarray,
    values: np.ndarray,
    maximum_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if maximum_points is None or len(time_s) <= maximum_points:
        return time_s, values
    if maximum_points < 4:
        raise ValueError("maximum_points must be at least 4")
    bins = max(1, (maximum_points - 2) // 2)
    edges = np.linspace(1, len(time_s) - 1, bins + 1, dtype=int)
    selected = {0, len(time_s) - 1}
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        segment = values[start:stop]
        selected.add(start + int(np.argmin(segment)))
        selected.add(start + int(np.argmax(segment)))
    indices = np.asarray(sorted(selected), dtype=int)
    return time_s[indices], values[indices]


def read_exl50u_shot_csv(
    path: str | Path,
    *,
    channels: Iterable[str] | None = None,
    t_start_s: float | None = -0.1,
    t_end_s: float | None = 0.1,
    maximum_points_per_signal: int | None = 4000,
    calibrations: Mapping[str, SignalCalibration] | None = None,
) -> EXL50UShot:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    try:
        shot = int(source.stem)
    except ValueError as exc:
        raise ValueError("EXL-50U shot CSV filename must be the numeric shot number") from exc
    requested = None if channels is None else {str(name) for name in channels}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("shot CSV is empty") from exc
        pairs = _column_pairs(header, shot)
        if requested is not None:
            unknown = requested - set(pairs)
            if unknown:
                raise ValueError(f"unknown requested channels: {', '.join(sorted(unknown))}")
            pairs = {name: pair for name, pair in pairs.items() if name in requested}
        time_lists = {name: [] for name in pairs}
        value_lists = {name: [] for name in pairs}
        for row in reader:
            for name, (time_index, data_index) in pairs.items():
                if max(time_index, data_index) >= len(row):
                    continue
                time_text = row[time_index].strip()
                value_text = row[data_index].strip()
                if not time_text or not value_text:
                    continue
                try:
                    time = float(time_text)
                    value = float(value_text)
                except ValueError:
                    continue
                if not (math.isfinite(time) and math.isfinite(value)):
                    continue
                if t_start_s is not None and time < t_start_s:
                    continue
                if t_end_s is not None and time > t_end_s:
                    continue
                time_lists[name].append(time)
                value_lists[name].append(value)

    supplied = dict(calibrations or {})
    signals: dict[str, ShotSignal] = {}
    warnings: list[str] = []
    for name in pairs:
        time = np.asarray(time_lists[name], dtype=float)
        values = np.asarray(value_lists[name], dtype=float)
        time, values = _extrema_downsample(time, values, maximum_points_per_signal)
        calibration = supplied.get(name, _default_calibration(name))
        signals[name] = ShotSignal(name, time, values, calibration)
        if calibration.status != "confirmed":
            warnings.append(f"{name}: calibration status is {calibration.status}")
    return EXL50UShot(shot, source.resolve(), signals, tuple(warnings))


def load_signal_calibrations(path: str | Path) -> dict[str, SignalCalibration]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    channels = payload.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise ValueError("calibration file must contain a non-empty channels mapping")
    return {
        str(name): SignalCalibration.from_mapping(str(name), values)
        for name, values in channels.items()
    }


def summarize_exl50u_shot_directory(
    directory: str | Path,
    *,
    t_start_s: float = -0.1,
    t_end_s: float = 0.1,
    calibrations: Mapping[str, SignalCalibration] | None = None,
) -> dict[str, object]:
    root = Path(directory)
    paths = sorted(root.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no shot CSV files found under {root}")
    core = {
        "ip", "HCN_NE001", "HA001", "I_CS", "LOOPV", "ITF01",
        "GAS_PRES01", "GAS_OUT01", "BT", "HXR001",
        *(f"I_PF{index}" for index in range(1, 11)),
    }
    shots = [
        read_exl50u_shot_csv(
            path,
            channels=core,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            maximum_points_per_signal=4000,
            calibrations=calibrations,
        ).summary()
        for path in paths
    ]
    return {
        "device": "EXL-50U",
        "source_directory": str(root.resolve()),
        "shot_count": len(shots),
        "window_s": [t_start_s, t_end_s],
        "shots": shots,
        "interpretation": {
            "equilibrium_usage": "none; these are diagnostic and coil waveforms",
            "pre_CFS_policy": "no GS/EFIT closed surfaces are inferred",
            "labels": "waveform-derived candidates requiring shot-log confirmation",
        },
    }


__all__ = [
    "SignalCalibration",
    "ShotSignal",
    "EXL50UShot",
    "read_exl50u_shot_csv",
    "load_signal_calibrations",
    "summarize_exl50u_shot_directory",
]
