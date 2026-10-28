"""Shot-resolved EXL-50U validation-set summaries and diagnostic PDF plots.

The report deliberately preserves each diagnostic's native time base.  Raw
signals are labelled as raw/relative when no confirmed calibration exists;
they are never silently converted to SI.  Waveform-derived success/failure
labels are candidate labels until an experiment log supplies their authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .experimental import (
    EXL50UShot,
    SignalCalibration,
    read_exl50u_shot_csv,
)


VALIDATION_CHANNELS = {
    "ip",
    "HCN_NE001",
    "HA001",
    "HXR001",
    "LOOPV",
    "BT",
    "I_CS",
    "GAS_PRES01",
    "GAS_OUT01",
    *(f"ECRH{index}_UK" for index in range(4)),
    *(f"I_PF{index}" for index in range(1, 11)),
}


def _display_values(shot: EXL50UShot, channel: str) -> tuple[np.ndarray, str, str]:
    signal = shot.require(channel)
    physical = signal.physical_values
    if physical is not None:
        return physical, str(signal.calibration.physical_unit), signal.calibration.status
    return signal.raw_values, signal.calibration.raw_unit, signal.calibration.status


def _relative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    baseline = float(np.nanpercentile(values, 5.0))
    scale = float(np.nanpercentile(np.abs(values - baseline), 99.0))
    return (values - baseline) / max(scale, 1.0e-30)


def _shot_payload(shot: EXL50UShot) -> dict[str, object]:
    label = shot.validation_label()
    channels: dict[str, object] = {}
    for name, signal in shot.signals.items():
        values, unit, status = _display_values(shot, name)
        if len(values):
            peak_index = int(np.argmax(np.abs(values)))
            peak = {
                "value": float(values[peak_index]),
                "time_s": float(signal.time_s[peak_index]),
            }
        else:
            peak = None
        channels[name] = {
            "available": True,
            "unit": unit,
            "calibration_status": status,
            "peak_absolute": peak,
            "sample_count": int(len(values)),
        }
    return {
        "shot": shot.shot,
        "candidate_label": label,
        "channels": channels,
        "missing_requested_channels": sorted(VALIDATION_CHANNELS - set(shot.signals)),
        "ece_status": "not_present_in_supplied_csv",
    }


def load_exl50u_validation_set(
    directory: str | Path,
    *,
    calibrations: Mapping[str, SignalCalibration] | None = None,
    t_start_s: float = -0.1,
    t_end_s: float = 0.1,
    maximum_points_per_signal: int | None = 4000,
) -> tuple[list[EXL50UShot], dict[str, object]]:
    root = Path(directory)
    paths = sorted(root.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no shot CSV files found under {root}")
    shots = [
        read_exl50u_shot_csv(
            path,
            # Read the available columns and report missing validation
            # channels explicitly.  Older exports need not contain every
            # diagnostic in the current validation contract.
            channels=None,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            maximum_points_per_signal=maximum_points_per_signal,
            calibrations=calibrations,
        )
        for path in paths
    ]
    labels = [str(shot.validation_label()["candidate"]) for shot in shots]
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "device": "EXL-50U",
        "source_directory": str(root.resolve()),
        "window_s": [float(t_start_s), float(t_end_s)],
        "shot_count": len(shots),
        "candidate_class_counts": {
            label: labels.count(label) for label in sorted(set(labels))
        },
        "shots": [_shot_payload(shot) for shot in shots],
        "quality_policy": {
            "label_authority": "waveform_heuristic_only; confirm against shot log",
            "pre_CFS_equilibrium": "none; no GS/EFIT surfaces are used",
            "ece": "unavailable in the supplied CSV files",
            "raw_channels": "plotted only as raw or relative values until calibration is confirmed",
            "prediction_claim": "mechanism and sensitivity validation only",
        },
    }
    return shots, payload


def _plot_channel(axis, shot: EXL50UShot, channel: str, *, relative: bool = False, label: str | None = None) -> None:
    if channel not in shot.signals:
        axis.text(0.5, 0.5, f"{channel}: unavailable", ha="center", va="center", transform=axis.transAxes)
        return
    signal = shot.require(channel)
    values, unit, status = _display_values(shot, channel)
    if relative:
        values = _relative(values)
        unit = "relative"
    axis.plot(signal.time_s * 1.0e3, values, lw=1.2, label=label or channel)
    axis.set_ylabel(f"{label or channel} [{unit}]\n{status}")
    axis.grid(alpha=0.25)


def write_exl50u_validation_pdf(
    shots: list[EXL50UShot],
    payload: Mapping[str, object],
    path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    color_by_label = {
        "startup_success_candidate": "#2a7f62",
        "termination_candidate": "#c64b40",
        "failed_or_inconclusive": "#8a6d3b",
        "unclassified": "#6b7280",
    }
    with PdfPages(output) as pdf:
        figure, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
        for shot in shots:
            label = str(shot.validation_label()["candidate"])
            color = color_by_label.get(label, "#6b7280")
            if "ip" in shot.signals:
                signal = shot.require("ip")
                current = signal.physical_values
                if current is not None:
                    axes[0].plot(signal.time_s * 1.0e3, current / 1.0e3, color=color, lw=1.2, label=f"{shot.shot} {label}")
            if "HCN_NE001" in shot.signals:
                signal = shot.require("HCN_NE001")
                density = signal.physical_values
                if density is not None:
                    axes[1].plot(signal.time_s * 1.0e3, density / 1.0e19, color=color, lw=1.2, label=str(shot.shot))
        axes[0].set_ylabel("Ip [kA]")
        axes[1].set_ylabel("HCN ne [1e19 m^-3]")
        axes[1].set_xlabel("time relative to Ip reference [ms]")
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].legend(fontsize=8, ncol=3)
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.suptitle("EXL-50U startup validation candidates: common-window comparison")
        figure.text(
            0.01,
            0.01,
            "Candidate labels require shot-log confirmation. No pre-CFS GS/EFIT surfaces are used. ECE is absent.",
            fontsize=8,
            color="#8b1e1e",
        )
        figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.95))
        pdf.savefig(figure, dpi=180)
        plt.close(figure)

        for shot in shots:
            label = shot.validation_label()
            figure, axes = plt.subplots(4, 2, figsize=(11.0, 8.5), sharex=True)
            flat = axes.ravel()
            _plot_channel(flat[0], shot, "ip", label="Ip")
            if "ip" in shot.signals and shot.require("ip").physical_values is not None:
                flat[0].lines[-1].set_ydata(shot.require("ip").physical_values / 1.0e3)
                flat[0].set_ylabel("Ip [kA]\n" + shot.require("ip").calibration.status)
                flat[0].relim()
                flat[0].autoscale_view()
            _plot_channel(flat[1], shot, "HCN_NE001", label="HCN ne")
            if "HCN_NE001" in shot.signals and shot.require("HCN_NE001").physical_values is not None:
                flat[1].lines[-1].set_ydata(shot.require("HCN_NE001").physical_values / 1.0e19)
                flat[1].set_ylabel("HCN ne [1e19 m^-3]\n" + shot.require("HCN_NE001").calibration.status)
                flat[1].relim()
                flat[1].autoscale_view()
            _plot_channel(flat[2], shot, "HA001", relative=True, label="H-alpha")
            _plot_channel(flat[3], shot, "HXR001", relative=True, label="HXR")
            _plot_channel(flat[4], shot, "LOOPV", label="loop voltage")
            ecrh_axis = flat[5]
            found_ecrh = False
            for index in range(4):
                channel = f"ECRH{index}_UK"
                if channel not in shot.signals:
                    continue
                signal = shot.require(channel)
                ecrh_axis.plot(signal.time_s * 1.0e3, signal.raw_values, lw=1.0, label=channel)
                found_ecrh = True
            if found_ecrh:
                ecrh_axis.legend(fontsize=7, ncol=2)
                ecrh_axis.set_ylabel("ECRH monitor [raw]\nun-calibrated")
                ecrh_axis.grid(alpha=0.25)
            else:
                ecrh_axis.text(0.5, 0.5, "ECRH monitors unavailable", ha="center", va="center", transform=ecrh_axis.transAxes)
            _plot_channel(flat[6], shot, "GAS_PRES01", label="gas pressure monitor")
            last = flat[7]
            handles = []
            labels = []
            if "BT" in shot.signals:
                signal = shot.require("BT")
                values, unit, status = _display_values(shot, "BT")
                line = last.plot(signal.time_s * 1.0e3, values, lw=1.0, color="#006d77")[0]
                handles.append(line)
                labels.append(f"BT [{unit}; {status}]")
                last.set_ylabel(f"BT [{unit}]")
            if "I_CS" in shot.signals:
                signal = shot.require("I_CS")
                values, unit, status = _display_values(shot, "I_CS")
                cs_axis = last.twinx()
                line = cs_axis.plot(signal.time_s * 1.0e3, values / 1.0e3, lw=1.0, color="#9c6644")[0]
                handles.append(line)
                labels.append(f"I_CS [kA; {status}]")
                cs_axis.set_ylabel("I_CS [kA]")
            if handles:
                last.legend(handles, labels, fontsize=7, loc="best")
            last.grid(alpha=0.25)
            for axis in axes[-1, :]:
                axis.set_xlabel("time [ms]")
            figure.suptitle(
                f"Shot {shot.shot}: {label['candidate']}  |  peak Ip={label.get('peak_Ip_A', 0.0) / 1e3:.1f} kA"
            )
            figure.text(
                0.01,
                0.012,
                "ECE: not present. Raw/relative traces are timing evidence only. Mechanism/sensitivity validation, not accurate Ip prediction.",
                fontsize=7.5,
                color="#8b1e1e",
            )
            figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.95))
            pdf.savefig(figure, dpi=180)
            plt.close(figure)
    return output


def build_exl50u_validation_artifacts(
    directory: str | Path,
    output_directory: str | Path,
    *,
    calibrations: Mapping[str, SignalCalibration] | None = None,
    t_start_s: float = -0.1,
    t_end_s: float = 0.1,
) -> dict[str, object]:
    shots, payload = load_exl50u_validation_set(
        directory,
        calibrations=calibrations,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "validation_set.json"
    pdf_path = output / "validation_diagnostics.pdf"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_exl50u_validation_pdf(shots, payload, pdf_path)
    return {**payload, "artifacts": {"json": str(json_path.resolve()), "pdf": str(pdf_path.resolve())}}


__all__ = [
    "VALIDATION_CHANNELS",
    "load_exl50u_validation_set",
    "write_exl50u_validation_pdf",
    "build_exl50u_validation_artifacts",
]
