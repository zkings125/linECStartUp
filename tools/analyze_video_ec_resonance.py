"""Exploratory 50 GHz EC resonance/camera comparison for Shot 16747.

This tool does not invert camera pixels into magnetic coordinates. It reports
the EFIT |B|=B_res layers at 102 and 500 ms and extracts camera-coordinate
brightness centroids/line profiles using the framewise timestamp table.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ecstartup.constants import E_CHARGE_C, M_E_KG
from ecstartup.physics.eqdsk import read_gfile


def _load_times(path: Path, start_ms: float, end_ms: float) -> list[float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        value = row.get("shot_time_ms_ocr", "")
        try:
            value = float(value)
        except ValueError:
            value = float(row["shot_time_ms_calibrated"])
        if start_ms - 5 <= value <= end_ms + 5:
            values.append(value)
    return values


def _extract_camera(video: Path, n: int, start_ms: float) -> list[np.ndarray]:
    video_start_s = (start_ms + 210.0) / 400.0
    with tempfile.TemporaryDirectory(prefix="shot16747_ec50_") as temp_dir:
        pattern = str(Path(temp_dir) / "%04d.png")
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{video_start_s:.6f}",
            "-i", str(video), "-vf", "fps=120,crop=1100:1100:0:0,scale=275:-1",
            "-frames:v", str(n), pattern,
        ], check=True)
        files = sorted(Path(temp_dir).glob("*.png"))
        return [np.asarray(Image.open(item).convert("L"), dtype=float) for item in files]


def analyze(video: Path, times_csv: Path, gfile_102: Path, gfile_500: Path, output: Path, start_ms: float, end_ms: float) -> None:
    times = _load_times(times_csv, start_ms, end_ms)
    frames = _extract_camera(video, len(times), start_ms)
    n = min(len(times), len(frames))
    times = np.asarray(times[:n])
    frames = frames[:n]
    stack = np.asarray(frames)
    baseline = np.percentile(stack[: max(1, min(5, n))], 50, axis=0)
    excess = np.clip(stack - baseline[None, ...], 0.0, None)
    vertical = excess.mean(axis=2)
    horizontal = excess.mean(axis=1)
    yy, xx = np.indices(excess.shape[1:])
    weights = excess.sum(axis=(1, 2)) + 1.0e-12
    centroid_x = (excess * xx[None, ...]).sum(axis=(1, 2)) / weights
    centroid_y = (excess * yy[None, ...]).sum(axis=(1, 2)) / weights
    output.mkdir(parents=True, exist_ok=True)
    with (output / "camera_intensity_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shot_time_ms", "excess_brightness_sum", "centroid_x_px", "centroid_y_px"])
        writer.writerows(zip(times, weights, centroid_x, centroid_y))

    f50 = read_gfile(gfile_500)
    f102 = read_gfile(gfile_102)
    B_res = 2.0 * np.pi * 50.0e9 * M_E_KG / E_CHARGE_C
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    for col, (g, title) in enumerate([(f102, "102 ms EFIT"), (f50, "500 ms EFIT")]):
        ax = fig.add_subplot(gs[0, col])
        br, bz, bp = g.field_arrays(); bmag = np.sqrt(br * br + bz * bz + bp * bp)
        ax.contour(g.R_grid_m, g.Z_grid_m, bmag.T, levels=[B_res], colors="tab:red", linewidths=1.8)
        ax.contour(g.R_grid_m, g.Z_grid_m, g.psi_Wb.T, levels=12, colors="0.45", linewidths=0.35)
        ax.plot(g.boundary_RZ_m[:, 0], g.boundary_RZ_m[:, 1], "k-", lw=1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="R [m]", ylabel="Z [m]", title=f"{title}: |B| = {B_res:.3f} T")
    ax = fig.add_subplot(gs[0, 2:])
    ax.axis("off")
    ax.text(0.02, 0.9, "50 GHz EC resonance interpretation", fontsize=13, weight="bold")
    ax.text(0.02, 0.68, "B_res = 1.786 T (fundamental)\nred contour: EFIT |B|=B_res\nblack contour: EFIT boundary\n\nCamera pixels are not mapped to R,Z here.\nTheir brightness may include EC emission,\nfield-line curvature, wall/limiter light and line-of-sight integration.", va="top", family="monospace")
    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(vertical.T, aspect="auto", origin="lower", extent=[times[0], times[-1], 0, vertical.shape[1]], cmap="magma")
    fig.colorbar(im, ax=ax, label="excess brightness")
    ax.set(xlabel="shot time [ms]", ylabel="camera vertical pixel", title="Framewise camera vertical brightness evolution")
    ax = fig.add_subplot(gs[2, 0:2])
    im = ax.imshow(horizontal, aspect="auto", origin="lower", extent=[0, horizontal.shape[1], times[0], times[-1]], cmap="magma")
    fig.colorbar(im, ax=ax, label="excess brightness")
    ax.set(xlabel="camera horizontal pixel", ylabel="shot time [ms]", title="Framewise camera horizontal brightness evolution")
    ax = fig.add_subplot(gs[2, 2:])
    ax.plot(times, centroid_x, label="brightness centroid x")
    ax.plot(times, centroid_y, label="brightness centroid y")
    ax.set(xlabel="shot time [ms]", ylabel="camera pixel", title="Brightness-centroid motion (camera coordinates)")
    ax.grid(alpha=0.25); ax.legend()
    fig.suptitle("Shot 16747: 50 GHz EC resonance and framewise camera comparison", fontsize=15)
    fig.savefig(output / "ec50_resonance_video_analysis.png", dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--times-csv", type=Path, required=True)
    p.add_argument("--gfile-102ms", type=Path, required=True)
    p.add_argument("--gfile-500ms", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/exl50u_16747_gfile"))
    p.add_argument("--start-ms", type=float, default=-200.0)
    p.add_argument("--end-ms", type=float, default=100.0)
    args = p.parse_args()
    analyze(args.video, args.times_csv, args.gfile_102ms, args.gfile_500ms, args.output, args.start_ms, args.end_ms)


if __name__ == "__main__":
    main()
