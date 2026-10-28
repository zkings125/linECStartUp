"""Track the visually clear vertical EC-like lines around -150--140 ms."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def analyze(frame_dir: Path, output: Path) -> None:
    files = sorted(frame_dir.glob("frame_*.png"))
    records = []
    images = []
    for path in files:
        image = np.asarray(Image.open(path).convert("L"), dtype=float)
        images.append(image)
        # The green label was read separately from the source frame; these four
        # labels are visible in the supplied -150--140 ms contact sheet.
    times = [-150.0, -147.0, -143.0, -140.0]
    left_cols = []
    right_cols = []
    for image, time_ms in zip(images, times):
        # Polar-view geometry: the central black vertical object is the center
        # post. Track the bright vertical lines immediately on its two sides.
        profile = image[280:720, 150:950].mean(axis=0)
        left = int(np.argmax(profile[250 - 150:500 - 150]) + 250)
        right = int(np.argmax(profile[500 - 150:800 - 150]) + 500)
        left_cols.append(left); right_cols.append(right)
        records.append((time_ms, left, right, float(profile[left - 150]), float(profile[right - 150])))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "clear_resonance_window_ridge.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shot_time_ms", "left_vertical_line_x_px", "right_vertical_line_x_px", "left_mean_brightness", "right_mean_brightness"])
        writer.writerows(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, image, time_ms, left, right in zip(axes.flat, images, times, left_cols, right_cols):
        ax.imshow(image, cmap="gray", origin="upper")
        ax.axvline(left, color="red", lw=1.5, label=f"left vertical line x={left} px")
        ax.axvline(right, color="cyan", lw=1.5, label=f"right vertical line x={right} px")
        ax.set_title(f"shot time {time_ms:g} ms")
        ax.set_xlim(100, 1000); ax.set_ylim(850, 300)
        ax.set_xlabel("camera x [px]"); ax.set_ylabel("camera y [px]")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Shot 16747: vertical bright-line evolution near -150 to -140 ms", fontsize=14)
    fig.savefig(output / "clear_resonance_window.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    analyze(Path("runs/exl50u_16747_gfile/frames_-150_-140"), Path("runs/exl50u_16747_gfile"))
