"""Analyze an EXL-50U EFIT gfile and make startup evolution figures.

This is a static-equilibrium analysis: the gfile supplies one magnetic
snapshot.  The startup time traces therefore use the existing ECStartup 0D
source/circuit model, while the reconstructed gfile field is held fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from ecstartup.config import SimulationConfig
from ecstartup.physics.eqdsk import read_gfile
from ecstartup.physics.fieldline import FieldLineTracer
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.workflow import run_simulation
from ecstartup.constants import E_CHARGE_C, M_E_KG


def _finite_json(value):
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def _config(gfile, frequency_hz: float, power_w: float, launch_R: float, launch_Z: float, t_end: float, pulse_end_s: float | None = None) -> SimulationConfig:
    a = max(gfile.equivalent_minor_radius_m, 1.0e-3)
    if pulse_end_s is None:
        pulse_end_s = min(0.001, t_end)
    pulse_end_s = min(max(float(pulse_end_s), 1.0e-7), t_end)
    return SimulationConfig.model_validate({
        "geometry": {
            "R0_m": gfile.rmaxis_m,
            "minor_radius_m": a,
            "volume_m3": max(gfile.volume_m3, 1.0e-9),
            "connection_length_m": max(1.0, 2.0 * np.pi * a),
            "B_t_T": max(abs(gfile.bcentr_T), 1.0e-6),
            "B_vertical_T": 0.0,
            "B_penetration_T": 1.0e-9,
            "R_bounds_m": [float(gfile.R_grid_m[0]), float(gfile.R_grid_m[-1])],
            "Z_bounds_m": [float(gfile.Z_grid_m[0]), float(gfile.Z_grid_m[-1])],
        },
        "gas": {"pressure_Pa": 0.05},
        "initial": {"ne_m3": 1.0e14, "ni_m3": 1.0e14, "Te_eV": 0.20, "Ti_eV": 0.03, "Ip_A": 0.0},
        "rf": {
            "enabled": True,
            "frequency_Hz": frequency_hz,
            "power_W": power_w,
            "absorption_fraction": 0.70,
            "pulse": {"start_s": 0.0, "end_s": pulse_end_s, "ramp_up_s": min(5.0e-5, 0.1 * pulse_end_s), "ramp_down_s": min(5.0e-5, 0.1 * pulse_end_s)},
            "launch": {"R_m": launch_R, "Z_m": launch_Z, "mode": "X2", "polarization": "X"},
            "hot_fraction": 0.20,
            "hot_creation_energy_eV": 100.0,
            "hot_loss_time_s": 2.0e-4,
            "hot_scattering_time_s": 5.0e-4,
            "orbit_asymmetry": 0.05,
            "current_drive_A_per_W": 0.0,
        },
        "circuit": {"loop_voltage_V": 5.0, "loop_voltage_ramp_s": 5.0e-4, "total_inductance_H": 2.0e-6},
        # This analysis is a schematic 0-100 ms study, not a convergence run.
        # Keep the core package defaults untouched and use a coarser, explicit
        # cadence here so a long diagnostic window remains practical.
        "numerics": {"t_end_s": t_end, "dt_s": min(1.0e-4, t_end / 300.0), "solver": "BDF"},
        "outputs": {"cadence_s": max(t_end / 300.0, 1.0e-7), "write_netcdf": True, "write_csv": True},
    })


def _stage_spans(times, diagnostics):
    spans = []
    start = 0
    for i in range(1, len(diagnostics) + 1):
        if i == len(diagnostics) or diagnostics[i]["stage"] != diagnostics[start]["stage"]:
            spans.append((times[start], times[i - 1], diagnostics[start]["stage"]))
            start = i
    return spans


def _video_montage(video_path: Path, output: Path, *, start_ms: float = -200.0, end_ms: float = 100.0, interval_ms: float | None = None) -> dict:
    """Extract every source frame in the requested shot-time window.

    The in-video green timestamp is the authority. ``interval_ms`` is retained
    only as an optional later down-sampling control; the default is frame-wise.
    """

    # The on-screen cursor gives t_shot[ms] ~= 400*t_video[s] - 210.
    video_start_s = (start_ms + 210.0) / 400.0
    # The on-screen calibration is t_shot ~= 0.4*t_video - 210 ms.  Thus a
    # 10 ms shot-time step is 25 ms of video time, i.e. 40 output samples/s
    # (about every 3 frames of the 120 fps source).
    sample_fps = 120.0 if interval_ms is None else 400.0 / interval_ms
    shot_step_ms = 400.0 / sample_fps
    count = int(round((end_ms - start_ms) / shot_step_ms)) + 3
    with tempfile.TemporaryDirectory(prefix="shot16747_frames_") as temp_dir:
        pattern = str(Path(temp_dir) / "%04d.png")
        label_pattern = str(Path(temp_dir) / "label_%04d.png")
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{video_start_s:.6f}",
            "-i", str(video_path), "-vf", f"fps={sample_fps:g},crop=1100:1100:0:0,scale=275:-1",
            "-frames:v", str(count), pattern,
        ], check=True)
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{video_start_s:.6f}",
            "-i", str(video_path), "-vf", f"fps={sample_fps:g},crop=500:80:500:0,scale=1000:160",
            "-frames:v", str(count), label_pattern,
        ], check=True)
        frame_files = sorted(Path(temp_dir).glob("*.png"))
        frames_all = [Image.open(p).convert("RGB") for p in frame_files if not p.name.startswith("label_")]
        labels = []
        label_files = sorted(Path(temp_dir).glob("label_*.png"))
        for label_file in label_files:
            try:
                text = subprocess.run(["tesseract", str(label_file), "stdout", "-l", "eng"], capture_output=True, text=True, check=False).stdout
                match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*s", text)
                labels.append(float(match.group(1)) * 1000.0 if match else None)
            except OSError:
                labels.append(None)
    if not frames_all:
        raise RuntimeError("视频未提取到帧")
    calibrated_all = [start_ms + i * (1000.0 / sample_fps * 0.4) for i in range(len(labels))]
    valid = [i for i, label in enumerate(labels) if start_ms - 5.0 <= (label if label is not None else calibrated_all[i]) <= end_ms + 5.0]
    frames = [frames_all[i] for i in valid if i < len(frames_all)]
    labels = [labels[i] for i in valid]
    calibrated = [calibrated_all[i] for i in valid]
    cols = 12
    rows = int(np.ceil(len(frames) / cols))
    tile_w, tile_h = frames[0].size
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "black")
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        x = (index % cols) * tile_w
        y = (index // cols) * tile_h
        canvas.paste(frame, (x, y))
        target_ms = calibrated[index]
        t_ms = labels[index] if labels[index] is not None else target_ms
        draw.rectangle((x + 4, y + 4, x + 88, y + 22), fill=(0, 0, 0))
        draw.text((x + 7, y + 6), f"{t_ms:g} ms", fill=(255, 255, 0))
    canvas.save(output)
    csv_path = output.with_name(output.stem + "_times.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "video_time_s", "shot_time_ms_ocr", "shot_time_ms_calibrated"])
        for index, label in enumerate(labels):
            writer.writerow([valid[index], video_start_s + valid[index] / sample_fps, label, calibrated[index]])
    return {
        "path": str(video_path),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "interval_ms": interval_ms,
        "video_fps": 120.0,
        "sampling_video_interval_ms": 1000.0 / sample_fps,
        "time_mapping": "t_shot_ms ~= 400*t_video_s - 210",
        "label_source": "OCR of green in-video time label; fallback to calibration if OCR fails",
        "label_times_ms": labels,
        "frame_count": len(frames),
        "times_csv": str(csv_path),
    }


def _plot_0_100ms_schematic(gfile, field, args, launch_R, launch_Z, B_res, output: Path, cfs_gfile=None) -> dict:
    """Make an explicitly schematic 0--100 ms topology/EC figure.

    No pre-CFS EFIT solution is implied.  The open-field stream function is a
    visual placeholder; every pre-100 ms contour is deliberately open.
    """

    t_end = 0.100
    times = np.linspace(0.0, t_end, 501, endpoint=False)
    cfs_time = max(float(args.cfs_time_ms) * 1.0e-3, t_end)
    width = max(float(args.cfs_transition_width_ms) * 1.0e-3, 1.0e-4)
    # By the user's constraint, no closed surface is shown before 100 ms.
    weights = np.zeros_like(times)
    R, Z = gfile.R_grid_m, gfile.Z_grid_m
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    # A poloidal flux sketch with dpsi/dR > 0. Its psi=const contours are
    # predominantly vertical open curves, matching the pre-CFS R-Z view.
    open_base = (RR - R.min()) / max(R.max() - R.min(), 1.0e-12)
    # The 500 ms snapshot places the fundamental 28 GHz resonance near this
    # major-radius column at the magnetic midplane. Before CFS, its actual
    # location is unknown, so we draw a separate open-layer placeholder.
    br_ref, bz_ref, bphi_ref = gfile.field_arrays()
    bmag_ref = np.sqrt(br_ref * br_ref + bz_ref * bz_ref + bphi_ref * bphi_ref)
    mid_z_index = int(np.argmin(np.abs(Z - gfile.zmaxis_m)))
    resonance_R = float(R[np.argmin(np.abs(bmag_ref[:, mid_z_index] - B_res))])
    snapshots_ms = np.arange(0.0, 100.0, 10.0)
    fig, axes = plt.subplots(3, 5, figsize=(18, 12), constrained_layout=True)
    levels = np.linspace(0.1, 0.9, 9)
    for ax, snapshot_ms in zip(axes[:2].flat, snapshots_ms):
        phase = float(snapshot_ms / 100.0)
        vertical_bow = 0.07 * phase * ((ZZ - gfile.zmaxis_m) / max(gfile.equivalent_minor_radius_m, 1.0e-6)) ** 2
        vertical_tilt = 0.018 * phase * ((ZZ - gfile.zmaxis_m) / max(gfile.equivalent_minor_radius_m, 1.0e-6))
        psi = open_base + vertical_bow + vertical_tilt
        ax.contour(R, Z, psi.T, levels=levels, cmap="viridis", linewidths=0.9)
        resonance_curve = resonance_R + 0.025 * phase + 0.035 * phase * ((Z - gfile.zmaxis_m) / max(gfile.equivalent_minor_radius_m, 1.0e-6)) ** 2
        ax.plot(resonance_curve, Z, "--", color="tab:red", lw=1.2, label="schematic EC resonance layer")
        ax.plot([gfile.R_grid_m[0], gfile.R_grid_m[-1], gfile.R_grid_m[-1], gfile.R_grid_m[0], gfile.R_grid_m[0]], [gfile.Z_grid_m[0], gfile.Z_grid_m[0], gfile.Z_grid_m[-1], gfile.Z_grid_m[-1], gfile.Z_grid_m[0]], "k--", lw=0.8, label="calculation window")
        ax.plot(launch_R, launch_Z, "*", color="tab:red", ms=9)
        ax.set_xlim(gfile.boundary_RZ_m[:, 0].min() - .08, gfile.boundary_RZ_m[:, 0].max() + .08)
        ax.set_ylim(gfile.boundary_RZ_m[:, 1].min() - .08, gfile.boundary_RZ_m[:, 1].max() + .08)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t = {snapshot_ms:g} ms, open-field sketch")
        ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    axes[0, 0].legend(fontsize=7, loc="best")

    # A schematic EC envelope: the actual 0D source remains independently
    # configurable; this line only visualizes the intended 0--100 ms timing.
    ec = np.ones_like(times) * args.power_w
    rise = min(0.005, 0.2 * cfs_time)
    ec *= np.clip(times / max(rise, 1.0e-6), 0.0, 1.0)
    fall_start = 0.095
    ec *= np.where(times <= fall_start, 1.0, np.clip((t_end - times) / (t_end - fall_start), 0.0, 1.0))
    axes[2, 0].plot(times * 1e3, ec / 1e3, color="tab:red", lw=2)
    axes[2, 0].set(xlabel="time [ms]", ylabel="EC input [kW]", title="EC injection envelope (schematic)")
    axes[2, 0].grid(alpha=0.25)
    axes[2, 1].plot(times * 1e3, weights, color="tab:green", lw=2, label="closed-surface fraction shown")
    axes[2, 1].axvline(100.0, color="k", ls="--", lw=1, label="end of pre-CFS window")
    if args.cfs_time_ms <= 100.0:
        axes[2, 1].axvline(args.cfs_time_ms, color="tab:red", ls=":", lw=1, label="user CFS estimate")
    axes[2, 1].set(xlabel="time [ms]", ylabel="0-1 weight", title="No closed surface shown for 0-100 ms")
    axes[2, 1].set_ylim(-0.03, 1.03); axes[2, 1].grid(alpha=0.25); axes[2, 1].legend(fontsize=8)
    if cfs_gfile is not None:
        axes[2, 2].axis("off")
        axes[2, 2].text(0.02, 0.92, "EC / magnetic reference", fontsize=12, weight="bold")
        axes[2, 2].text(0.02, 0.78, f"flat-top gfile: 500 ms\nshot: 16747\nfrequency: {args.frequency_hz/1e9:.3g} GHz\nharmonic: {args.harmonic:g}\nB_res: {B_res:.4g} T\nB_launch: {field.evaluate(launch_R, launch_Z, 0.0).magnitude_T:.4g} T\nCFS reference: 102 ms", va="top", family="monospace")
        ax = axes[2, 3]
        levels_cfs = np.linspace(float(np.nanmin(cfs_gfile.psi_Wb)), float(np.nanmax(cfs_gfile.psi_Wb)), 16)
        ax.contour(cfs_gfile.R_grid_m, cfs_gfile.Z_grid_m, cfs_gfile.psi_Wb.T, levels=levels_cfs, linewidths=0.45, cmap="viridis")
        ax.plot(cfs_gfile.boundary_RZ_m[:, 0], cfs_gfile.boundary_RZ_m[:, 1], "r-", lw=1.8, label="EFIT boundary")
        ax.plot(cfs_gfile.rmaxis_m, cfs_gfile.zmaxis_m, "ko", ms=4, label="magnetic axis")
        ax.plot(launch_R, launch_Z, "*", color="tab:red", ms=8, label="EC launch")
        ax.set_xlim(cfs_gfile.boundary_RZ_m[:, 0].min() - .08, cfs_gfile.boundary_RZ_m[:, 0].max() + .08)
        ax.set_ylim(cfs_gfile.boundary_RZ_m[:, 1].min() - .08, cfs_gfile.boundary_RZ_m[:, 1].max() + .08)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="R [m]", ylabel="Z [m]", title="102 ms first closed-surface EFIT reference")
        ax.legend(fontsize=7, loc="best")
    else:
        axes[2, 3].axis("off")
        axes[2, 3].text(0.02, 0.92, "EC / magnetic reference", fontsize=12, weight="bold")
        axes[2, 3].text(0.02, 0.78, f"gfile snapshot: 500 ms\nshot: 16747\nfrequency: {args.frequency_hz/1e9:.3g} GHz\nharmonic: {args.harmonic:g}\nB_res: {B_res:.4g} T\nB_launch: {field.evaluate(launch_R, launch_Z, 0.0).magnitude_T:.4g} T\nCFS estimate: {args.cfs_time_ms:g} ms", va="top", family="monospace")
        axes[2, 3].text(0.02, 0.25, "Pre-CFS panels contain no closed magnetic contour.\n500 ms EFIT is used only as a flat-top reference.\nA 102 ms gfile is required for a real final startup surface.", va="top", color="tab:red")
    axes[2, 4].axis("off")
    axes[2, 4].text(0.02, 0.92, "Video time base", fontsize=12, weight="bold")
    axes[2, 4].text(0.02, 0.78, "Shot16747.mp4: 120 fps, 423 frames\nvisible camera start: about -210 ms\nframe-to-shot mapping: t_exp ~= 0.4 t_video - 210 ms\npre-CFS schematic snapshots: every 10 ms", va="top", family="monospace")
    fig.suptitle("EXL-50U 0-100 ms startup schematic (500 ms gfile used only as flat-top parameter reference)", fontsize=14)
    fig.savefig(output / "exl50u_16747_0_100ms_startup_schematic.png", dpi=180)
    plt.close(fig)
    return {
        "schematic_time_window_s": [0.0, t_end],
        "assumed_cfs_time_s": cfs_time,
        "transition_width_s": width,
        "gfile_reference_time_s": 0.5,
        "cfs_gfile_reference": str(cfs_gfile.path) if cfs_gfile is not None else None,
        "cfs_gfile_time_s": 0.102 if cfs_gfile is not None else None,
        "pre_cfs_field_status": "open_field_only_no_closed_contours",
        "pre_cfs_RZ_orientation": "predominantly_vertical_open_psi_contours_dpsi_dR_positive",
        "ec_resonance_layer_status": "separate_open_layer_placeholder_not_pre_cfs_EFIT",
        "ec_resonance_reference_R_m": resonance_R,
        "snapshot_interval_ms": 10.0,
        "ec_envelope_status": "schematic_timing_envelope",
    }


def analyze(args) -> Path:
    gfile = read_gfile(args.gfile)
    cfs_gfile = read_gfile(args.cfs_gfile) if args.cfs_gfile is not None else None
    launch_R = args.launch_R if args.launch_R is not None else float(np.max(gfile.boundary_RZ_m[:, 0]) - 0.03)
    launch_Z = args.launch_Z if args.launch_Z is not None else gfile.zmaxis_m
    config = _config(
        gfile, args.frequency_hz, args.power_w, launch_R, launch_Z, args.t_end_s,
        pulse_end_s=min(0.095, args.t_end_s),
    )
    field = gfile.to_field_provider(boundary="clip", configuration="EXL50U_16747_gfile")
    model = GlobalStartupModel(config, magnetic_field=field)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = run_simulation(config, output / "startup_run", model=model)

    R = gfile.R_grid_m
    Z = gfile.Z_grid_m
    BR, BZ, Bphi = gfile.field_arrays()
    Bmag = np.sqrt(BR * BR + BZ * BZ + Bphi * Bphi)
    harmonic = float(args.harmonic)
    B_res = 2.0 * np.pi * args.frequency_hz * M_E_KG / E_CHARGE_C / harmonic
    launch_point = field.evaluate(launch_R, launch_Z, gfile.current_A, 0.0)
    traces = []
    tracer = FieldLineTracer(field, bounds=config.geometry, step_m=args.fieldline_step_m, max_length_m=args.fieldline_max_length_m)
    for R0, Z0 in [(launch_R, launch_Z), (gfile.rmaxis_m, gfile.zmaxis_m), (gfile.rmaxis_m + 0.5 * gfile.equivalent_minor_radius_m, gfile.zmaxis_m)]:
        trace = tracer.trace(R0, Z0, Ip_A=gfile.current_A, direction="both")
        traces.append(trace)

    summary = gfile.summary()
    summary.update({
        "analysis_output": str(output),
        "ec_frequency_Hz": args.frequency_hz,
        "ec_power_W": args.power_w,
        "ec_harmonic": harmonic,
        "ec_resonance_B_T": B_res,
        "ec_launch_R_m": launch_R,
        "ec_launch_Z_m": launch_Z,
        "ec_launch_B_R_T": launch_point.B_R_T,
        "ec_launch_B_Z_T": launch_point.B_Z_T,
        "ec_launch_B_phi_T": launch_point.B_phi_T,
        "ec_launch_B_T": launch_point.magnitude_T,
        "ec_launch_resonance_ratio": launch_point.magnitude_T / max(B_res, 1.0e-30),
        "field_B_min_T": float(np.min(Bmag)),
        "field_B_max_T": float(np.max(Bmag)),
        "startup_success": bool(result.success),
        "startup_event_count": len(result.events),
        "startup_final_stage": result.diagnostics[-1]["stage"],
        "fieldline_traces": [trace.to_dict(include_path=False) for trace in traces],
    })
    if cfs_gfile is not None:
        summary["cfs_gfile_summary"] = cfs_gfile.summary()
    schematic_meta = _plot_0_100ms_schematic(gfile, field, args, launch_R, launch_Z, B_res, output, cfs_gfile=cfs_gfile)
    if args.video is not None:
        schematic_meta["video_montage"] = _video_montage(args.video, output / "shot16747_camera_-200_100ms_framewise_montage.png")
    summary["startup_schematic"] = schematic_meta
    (output / "gfile_summary.json").write_text(json.dumps(_finite_json(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    times = np.asarray(result.times_s)
    d = result.diagnostics
    def series(name): return np.asarray([float(row.get(name, np.nan)) for row in d])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax = axes[0, 0]
    levels = np.linspace(float(np.nanmin(gfile.psi_Wb)), float(np.nanmax(gfile.psi_Wb)), 24)
    cs = ax.contour(R, Z, gfile.psi_Wb.T, levels=levels, linewidths=0.45, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.3f")
    ax.plot(gfile.boundary_RZ_m[:, 0], gfile.boundary_RZ_m[:, 1], "r-", lw=1.8, label="EFIT boundary")
    ax.plot(gfile.limiter_RZ_m[:, 0], gfile.limiter_RZ_m[:, 1], "k--", lw=1.0, label="limiter")
    ax.plot(gfile.rmaxis_m, gfile.zmaxis_m, "wo", mec="k", ms=6, label="magnetic axis")
    ax.plot(launch_R, launch_Z, "*", color="tab:red", ms=11, label="EC launch")
    ax.set(xlabel="R [m]", ylabel="Z [m]", title="500 ms flat-top EFIT reference (not startup topology)")
    ax.set_aspect("equal", adjustable="box"); ax.legend(fontsize=7, loc="best")
    ax = axes[0, 1]
    im = ax.pcolormesh(R, Z, Bmag.T, shading="auto", cmap="magma")
    cb = fig.colorbar(im, ax=ax); cb.set_label("|B| [T]")
    ax.contour(R, Z, Bmag.T, levels=[B_res], colors="cyan", linewidths=1.4)
    ax.plot(launch_R, launch_Z, "w*", ms=10)
    ax.set(xlabel="R [m]", ylabel="Z [m]", title=f"|B| and EC resonance B={B_res:.4g} T")
    ax.set_xlim(gfile.boundary_RZ_m[:, 0].min() - .05, gfile.boundary_RZ_m[:, 0].max() + .05); ax.set_ylim(gfile.boundary_RZ_m[:, 1].min() - .05, gfile.boundary_RZ_m[:, 1].max() + .05)
    ax = axes[0, 2]
    colors = ["tab:red", "tab:blue", "tab:green"]
    for trace, color in zip(traces, colors):
        for branch in trace.branches:
            ax.plot(branch.path_R_m, branch.path_Z_m, color=color, lw=0.8)
    ax.plot(gfile.boundary_RZ_m[:, 0], gfile.boundary_RZ_m[:, 1], "k-", lw=1.2)
    ax.plot(launch_R, launch_Z, "r*", ms=10)
    ax.set(xlabel="R [m]", ylabel="Z [m]", title="Static-gfile field-line traces")
    ax.set_aspect("equal", adjustable="box")
    ax = axes[1, 0]
    ax.semilogy(times * 1e3, np.maximum(series("ne_m3"), 1.0), label=r"$n_e$")
    ax.semilogy(times * 1e3, np.maximum(series("n_hot_m3"), 1.0), label=r"$n_{hot}$")
    ax.set(xlabel="time [ms]", ylabel="density [m$^{-3}$]", title="Breakdown / preionization density")
    ax.grid(alpha=0.25); ax.legend()
    ax = axes[1, 1]
    ax.plot(times * 1e3, series("Te_eV"), label=r"$T_e$")
    ax.plot(times * 1e3, series("Ip_A") / 1e3, label=r"$I_p$ [kA]")
    ax.set(xlabel="time [ms]", ylabel="eV or kA", title="Electron temperature and plasma current")
    ax.grid(alpha=0.25); ax.legend()
    ax = axes[1, 2]
    ax.plot(times * 1e3, series("P_rf_W_m3") / 1e6, label="absorbed EC power density [MW/m3]")
    ax.plot(times * 1e3, series("I_rf_A") / 1e3, label="I_RF [kA]")
    ax.plot(times * 1e3, series("I_tail_A") / 1e3, label="I_tail [kA]")
    ax.plot(times * 1e3, series("I_vertical_charge_A") / 1e3, label="I_vertical_charge [kA]")
    ax.set(xlabel="time [ms]", ylabel="power density / current", title="EC input and current mechanisms")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    for axis in axes.flat:
        for t0, t1, stage in _stage_spans(times, d):
            if stage != "vacuum":
                axis.axvspan(t0 * 1e3, t1 * 1e3, alpha=0.035, color="tab:orange" if stage == "breakdown" else "tab:green")
    fig.suptitle("EXL-50U shot 16747: gfile-constrained EC startup schematic", fontsize=14)
    fig.savefig(output / "exl50u_16747_startup_evolution.png", dpi=180)
    plt.close(fig)

    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gfile", type=Path)
    parser.add_argument("--cfs-gfile", type=Path, default=None, help="optional EFIT gfile just after CFS (shown only as a separate reference panel)")
    parser.add_argument("--video", type=Path, default=None, help="optional shot video for a camera-only 10 ms contact sheet")
    parser.add_argument("--output", type=Path, default=Path("runs/exl50u_16747_gfile"))
    parser.add_argument("--frequency-hz", type=float, default=28.0e9)
    parser.add_argument("--power-w", type=float, default=20.0e3)
    parser.add_argument("--harmonic", type=float, default=1.0)
    parser.add_argument("--launch-R", type=float, default=None)
    parser.add_argument("--launch-Z", type=float, default=None)
    parser.add_argument("--t-end-s", type=float, default=0.100)
    parser.add_argument("--fieldline-step-m", type=float, default=5.0e-3)
    parser.add_argument("--fieldline-max-length-m", type=float, default=20.0)
    parser.add_argument("--cfs-time-ms", type=float, default=85.0, help="assumed CFS transition time for the 0-100 ms schematic")
    parser.add_argument("--cfs-transition-width-ms", type=float, default=3.0, help="smooth transition width for the schematic")
    args = parser.parse_args()
    output = analyze(args)
    print(json.dumps({"output": str(output), "summary": str(output / 'gfile_summary.json'), "figure": str(output / 'exl50u_16747_startup_evolution.png')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
