"""Run editable cross-device startup inputs and build an audited PDF report.

The cases intentionally share the same global equations. Device files change
only explicit geometry, field configuration, gas, RF and circuit inputs. The
results are diagnostic mechanism/sensitivity calculations until each device
passes the shot-resolved validation gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ecstartup.config import SimulationConfig, dump_resolved_config, load_config
from ecstartup.constants import E_CHARGE_C, K_B_J_K, M_E_KG, PI
from ecstartup.solver import integrate


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "devices"
RUN_DIR = ROOT / "runs" / "cross_device_startup"
FIG_DIR = ROOT / "output" / "figures" / "cross_device_startup"
PDF_PATH = ROOT / "output" / "pdf" / "cross_device_ec_startup_report.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\Deng.ttf")


@dataclass(frozen=True)
class DeviceSpec:
    key: str
    config_name: str
    category: str
    experiment_class: str
    doi: str
    evidence: str
    literature_anchor: tuple[float, float] | None = None


DEVICES = (
    DeviceSpec(
        "MAST",
        "mast_ebw_startup.ini",
        "spherical",
        "solenoid-free EBW startup",
        "10.1088/0029-5515/50/2/022004",
        "100 kW -> about 17 kA; optimized vertical-field ramp -> about 33 kA",
        (100.0e3, 17.0e3),
    ),
    DeviceSpec(
        "SUNIST",
        "sunist_ec_startup.ini",
        "spherical",
        "non-inductive 2.45 GHz startup",
        "10.1088/0029-5515/51/6/063021",
        "R0/a=0.30/0.23 m, Bt<0.15 T; pressure-sensitive transient",
    ),
    DeviceSpec(
        "LATE",
        "late_ech_startup.ini",
        "spherical",
        "ECH equilibrium formation",
        "10.1088/0029-5515/45/11/026",
        "low-aspect-ratio equilibrium formation by ECH",
    ),
    DeviceSpec(
        "QUEST",
        "quest_ec_startup.ini",
        "spherical",
        "EC startup after retarding-field cancellation",
        "10.1088/1741-4326/ad6914",
        "Te about 1 keV at ne>1e18 m^-3; stronger X2 single-pass absorption",
    ),
    DeviceSpec(
        "VEST",
        "vest_tpc_startup.ini",
        "spherical",
        "TPC ECH-assisted low-voltage startup",
        "10.1088/0029-5515/57/1/016001",
        "TPC broadens pressure/power window and reduces loop-voltage demand",
    ),
    DeviceSpec(
        "DIII-D",
        "diiid_ech_assisted_startup.ini",
        "conventional",
        "low-voltage Ohmic + ECH assisted",
        "10.1088/0029-5515/31/11/001",
        "ECH-assisted inductive startup; not pure EC current startup",
    ),
    DeviceSpec(
        "J-TEXT",
        "jtext_no_cs_startup.ini",
        "conventional",
        "representative no-CS EC attempt",
        "10.1051/epjconf/202431302003",
        "105 GHz / 500 kW system scale; not a reconstructed failed shot",
    ),
)


MECHANISM_CN = {
    "ohmic": "CS/感应",
    "direct_rf": "直接 RF",
    "kinetic": "F-P EC",
    "hot_tail": "热尾优先约束",
    "vertical_charge": "垂直电荷分离",
    "none": "无可分辨驱动",
    "not available": "无数据",
}
PHASE_EDGES = (0.0, 0.20, 0.50, 1.0)
PHASE_LABELS = ("前20%", "中间30%", "后50%")


def setup() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "resolved_inputs").mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams.update(
            {"font.family": family, "axes.unicode_minus": False, "font.size": 9.0}
        )
        pdfmetrics.registerFont(TTFont("ReportCN", str(FONT_PATH)))
    else:
        plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
        pdfmetrics.registerFont(TTFont("ReportCN", font_manager.findfont("DejaVu Sans")))


def _power_config(base: SimulationConfig, power_W: float) -> SimulationConfig:
    raw = base.model_dump(mode="python")
    source_name = base.threshold_scan.source_name
    matched = False
    for source in raw["rf"]["sources"]:
        enabled = source["name"] == source_name if source_name else not matched
        source["enabled"] = enabled
        if enabled:
            source["power_W"] = float(power_W)
            matched = True
    if not matched:
        raise ValueError(f"No RF source selected by threshold_scan.source_name={source_name!r}")
    raw["outputs"]["write_pdf_report"] = False
    raw["outputs"]["write_current_profile"] = False
    raw["outputs"]["write_csv"] = False
    raw["outputs"]["write_netcdf"] = False
    return SimulationConfig.model_validate(raw)


def _event_time(result, event_type: str) -> float | None:
    return next((float(item.time_s) for item in result.events if item.type == event_type), None)


def _phase_windows(config: SimulationConfig) -> tuple[tuple[float, float], ...]:
    start = float(config.numerics.t_start_s)
    duration = float(config.numerics.t_end_s - config.numerics.t_start_s)
    edges = tuple(start + fraction * duration for fraction in PHASE_EDGES)
    return tuple(zip(edges[:-1], edges[1:]))


def _dominant_in_window(result, left_s: float, right_s: float) -> str:
    selected: list[dict[str, Any]] = []
    final_window = math.isclose(right_s, float(result.times_s[-1]))
    for time_s, row in zip(result.times_s, result.diagnostics):
        include = left_s <= float(time_s) <= right_s if final_window else left_s <= float(time_s) < right_s
        if include:
            selected.append(row)
    if not selected:
        return "not available"
    component_names = {
        "ohmic": "I_ohmic_A",
        "direct_rf": "I_rf_A",
        "kinetic": "I_kinetic_A",
        "hot_tail": "I_tail_A",
        "vertical_charge": "I_vertical_charge_A",
    }
    means = {
        name: float(np.mean([abs(float(row.get(field, 0.0))) for row in selected]))
        for name, field in component_names.items()
    }
    if max(means.values(), default=0.0) <= 1.0e-9:
        return "none"
    return max(means, key=means.get)


def _resonance_geometry(config: SimulationConfig) -> dict[str, float | bool]:
    source = next(source for source in config.rf.sources if source.enabled)
    resonance_B = (2.0 * PI * source.frequency_Hz * M_E_KG / E_CHARGE_C) / source.harmonic
    resonance_R = config.geometry.B_t_T * config.geometry.R0_m / resonance_B
    inner = config.geometry.R0_m - config.geometry.minor_radius_m
    outer = config.geometry.R0_m + config.geometry.minor_radius_m
    return {
        "resonance_B_T": resonance_B,
        "resonance_R_m": resonance_R,
        "resonance_in_geometric_cross_section": bool(inner <= resonance_R <= outer),
    }


def _case_summary(power_W: float, result, config: SimulationConfig) -> dict[str, Any]:
    rows = result.diagnostics
    peak_index = int(np.argmax([abs(float(row["Ip_A"])) for row in rows]))
    peak = rows[peak_index]
    final = rows[-1]
    initial_neutral = config.gas.pressure_Pa / (K_B_J_K * config.gas.Tn_K)
    return {
        "power_W": float(power_W),
        "solver_success": bool(result.success),
        "peak_Ip_A": float(peak["Ip_A"]),
        "peak_time_s": float(result.times_s[peak_index]),
        "final_Ip_A": float(final["Ip_A"]),
        "peak_ne_m3": max(float(row["ne_m3"]) for row in rows),
        "peak_Te_eV": max(float(row["Te_eV"]) for row in rows),
        "peak_closed_fraction": max(float(row["closed_fraction"]) for row in rows),
        "peak_rf_absorbed_W": max(float(row["rf_power_absorbed_W"]) for row in rows),
        "peak_Da_ion": max(
            float(row["nu_ion_s"]) / max(float(row["nu_loss_s"]), 1.0e-30)
            for row in rows
        ),
        "power_density_W_m3": float(power_W) / config.geometry.resolved_volume_m3,
        "power_per_initial_neutral_W": float(power_W)
        / max(initial_neutral * config.geometry.resolved_volume_m3, 1.0),
        "final_stage": str(final.get("stage", "unknown")),
        "breakdown_time_s": _event_time(result, "breakdown"),
        "preionization_time_s": _event_time(result, "preionization"),
        "cfs_time_s": _event_time(result, "cfs"),
        "dominant_mechanisms": [
            _dominant_in_window(result, left, right) for left, right in _phase_windows(config)
        ],
        "time_s": [float(value) for value in result.times_s],
        "Ip_A": [float(row["Ip_A"]) for row in rows],
        "ne_m3": [float(row["ne_m3"]) for row in rows],
        "Te_eV": [float(row["Te_eV"]) for row in rows],
    }


def _failure_explanation(case: dict[str, Any], config: SimulationConfig) -> str:
    if case["cfs_time_s"] is not None:
        return "达到解析 CFS 候选判据；仍需真实场线/自由边界平衡确认。"
    if case["preionization_time_s"] is not None:
        return "已达预电离，但 Ip 自场未超过外加垂直场+穿透场阈值。"
    if case["breakdown_time_s"] is not None:
        return "达到击穿密度，但温度/密度未持续进入预电离电流爬升阶段。"
    if case["peak_Da_ion"] < 1.0:
        return "电离 Damkohler 数低于 1：开场线损失快于体电离增长。"
    if config.circuit.cs_mode == "without_cs":
        return "无 CS 且波粒动量闭合不足；热尾代理不能建立足够 Ip。"
    return "即使有低环压，当前吸收/电离与约束组合仍未通过持续击穿判据。"


def run_matrix() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for spec in DEVICES:
        config_path = CONFIG_DIR / spec.config_name
        config = load_config(config_path)
        dump_resolved_config(config, RUN_DIR / "resolved_inputs" / f"{spec.key}.json")
        geometry = config.geometry
        resonance = _resonance_geometry(config)
        cases = []
        for power_W in config.threshold_scan.ec_powers_W:
            case_config = _power_config(config, power_W)
            result = integrate(case_config)
            cases.append(_case_summary(power_W, result, case_config))
        baseline_peak = float(cases[0]["peak_Ip_A"])
        for case in cases:
            case["ec_attributable_peak_increment_A"] = float(case["peak_Ip_A"]) - baseline_peak
            case["outcome"] = (
                "CFS candidate"
                if case["cfs_time_s"] is not None
                else "preionization"
                if case["preionization_time_s"] is not None
                else "breakdown"
                if case["breakdown_time_s"] is not None
                else "transient breakdown-density crossing"
                if case["peak_ne_m3"] >= config.thresholds.breakdown_density_m3
                else "no sustained breakdown"
            )
        highest = cases[-1]
        devices.append(
            {
                "name": spec.key,
                "category": spec.category,
                "experiment_class": spec.experiment_class,
                "doi": spec.doi,
                "evidence": spec.evidence,
                "config_file": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                "validation_status": config.validation.intended_use,
                "startup_window": config.startup_window.describe(
                    t_start_s=config.numerics.t_start_s,
                    t_end_s=config.numerics.t_end_s,
                ),
                "geometry": {
                    "R0_m": geometry.R0_m,
                    "minor_radius_m": geometry.minor_radius_m,
                    "aspect_ratio": geometry.R0_m / geometry.minor_radius_m,
                    "B_t_T": geometry.B_t_T,
                    "B_vertical_T": geometry.B_vertical_T,
                    "connection_length_m": geometry.connection_length_m,
                    "volume_m3": geometry.resolved_volume_m3,
                    "field_configuration": geometry.field_configuration,
                },
                "rf": {
                    "frequency_Hz": next(source.frequency_Hz for source in config.rf.sources if source.enabled),
                    "harmonic": next(source.harmonic for source in config.rf.sources if source.enabled),
                    "cs_mode": config.circuit.cs_mode,
                },
                "gas_pressure_Pa": config.gas.pressure_Pa,
                "resonance": resonance,
                "literature_anchor": (
                    {"power_W": spec.literature_anchor[0], "Ip_A": spec.literature_anchor[1]}
                    if spec.literature_anchor
                    else None
                ),
                "highest_power_failure_explanation": _failure_explanation(highest, config),
                "cases": cases,
            }
        )
    payload = {
        "model_status": "diagnostic_only",
        "time_window_policy": "per-device configured interval and startup_window semantics",
        "pre_cfs_equilibrium_policy": "No GS/EFIT closed flux surfaces are used.",
        "devices": devices,
    }
    (RUN_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return devices


def _style_axes(axes) -> None:
    for axis in np.atleast_1d(axes).flat:
        axis.grid(True, alpha=0.25)


def plot_geometry(devices: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    colors_by_category = {"spherical": "#147d73", "conventional": "#b34e3a"}
    label_offsets = {
        "MAST": (4, 8),
        "SUNIST": (8, -10),
        "LATE": (-26, -10),
        "QUEST": (4, 8),
        "VEST": (-24, 14),
        "DIII-D": (4, 6),
        "J-TEXT": (4, 6),
    }
    for device in devices:
        g = device["geometry"]
        color = colors_by_category[device["category"]]
        axes[0].scatter(g["aspect_ratio"], g["B_t_T"], s=70 + 150 * g["minor_radius_m"], color=color)
        axes[0].annotate(
            device["name"],
            (g["aspect_ratio"], g["B_t_T"]),
            xytext=label_offsets[device["name"]],
            textcoords="offset points",
            fontsize=8,
        )
        resonance = device["resonance"]
        inner = g["R0_m"] - g["minor_radius_m"]
        outer = g["R0_m"] + g["minor_radius_m"]
        axes[1].plot([inner, outer], [device["name"], device["name"]], color="#9da8ad", lw=6, solid_capstyle="butt")
        marker = "o" if resonance["resonance_in_geometric_cross_section"] else "x"
        axes[1].scatter(resonance["resonance_R_m"], device["name"], marker=marker, s=65, color=color, zorder=3)
    axes[0].set(xlabel="aspect ratio A=R0/a", ylabel="Bt [T]", title="装置尺寸与环向场")
    axes[0].margins(x=0.16, y=0.10)
    axes[1].set(xlabel="major radius R [m]", title="真空场谐波层与几何截面")
    axes[1].text(0.02, 0.02, "圆点: 层位于 R0±a；叉号: 简化几何下不可达", transform=axes[1].transAxes, fontsize=8)
    _style_axes(axes)
    fig.suptitle("跨装置显式输入参数（不是启动前平衡重建）", fontsize=13, weight="bold")
    output = FIG_DIR / "device_geometry_and_resonance.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_traces(devices: list[dict[str, Any]], category: str, filename: str) -> Path:
    selected = [device for device in devices if device["category"] == category]
    columns = 2
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(10.8, 3.3 * rows), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for axis, device in zip(axes.flat, selected):
        cases = device["cases"]
        for index, case in enumerate(cases):
            t_ms = np.asarray(case["time_s"]) * 1.0e3
            axis.plot(t_ms, np.asarray(case["Ip_A"]) / 1.0e3, color=cmap(index / max(len(cases) - 1, 1)), label=f"{case['power_W']/1e3:g} kW")
        axis.set(title=f"{device['name']} | {device['rf']['cs_mode']}", xlabel="time [ms]", ylabel="Ip surrogate [kA]")
        axis.legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    _style_axes(axes)
    label = "球形环/低纵横比" if category == "spherical" else "常规托卡马克"
    fig.suptitle(f"{label}: 各装置配置启动时间窗内的电流演化", fontsize=13, weight="bold")
    output = FIG_DIR / filename
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_power_response(devices: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.3), squeeze=False, constrained_layout=True)
    for axis, device in zip(axes.flat, devices):
        powers = np.asarray([case["power_W"] for case in device["cases"]]) / 1.0e3
        peak = np.asarray([abs(case["peak_Ip_A"]) for case in device["cases"]]) / 1.0e3
        increment = np.asarray([abs(case["ec_attributable_peak_increment_A"]) for case in device["cases"]]) / 1.0e3
        axis.plot(powers, peak, "o-", color="#22577a", label="total peak")
        axis.plot(powers, increment, "s--", color="#d17a22", label="EC increment")
        anchor = device["literature_anchor"]
        if anchor:
            axis.scatter(anchor["power_W"] / 1.0e3, anchor["Ip_A"] / 1.0e3, marker="*", s=120, color="#9b2226", label="literature")
        axis.set(title=device["name"], xlabel="injected EC [kW]", ylabel="peak current [kA]")
        axis.legend(frameon=False, fontsize=7)
    for axis in axes.flat[len(devices):]:
        axis.axis("off")
    _style_axes(axes)
    fig.suptitle("功率-电流响应：统一方程，不按装置拟合 A/W", fontsize=13, weight="bold")
    output = FIG_DIR / "power_current_response.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_outcomes(devices: list[dict[str, Any]]) -> Path:
    outcome_code = {
        "no sustained breakdown": 0,
        "transient breakdown-density crossing": 1,
        "breakdown": 2,
        "preionization": 3,
        "CFS candidate": 4,
    }
    mechanism_order = ["none", "ohmic", "direct_rf", "kinetic", "hot_tail", "vertical_charge", "not available"]
    outcome = np.asarray([[outcome_code[case["outcome"]] for case in device["cases"]] for device in devices], dtype=float)
    mechanisms = np.asarray(
        [[mechanism_order.index(name) for name in device["cases"][-1]["dominant_mechanisms"]] for device in devices],
        dtype=float,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    left = axes[0].imshow(
        outcome,
        aspect="auto",
        cmap=matplotlib.colors.ListedColormap(
            ["#d9d9d9", "#f4a261", "#e9c46a", "#70a1d7", "#2a9d8f"]
        ),
        vmin=-0.5,
        vmax=4.5,
    )
    axes[0].set(yticks=np.arange(len(devices)), yticklabels=[d["name"] for d in devices], xticks=np.arange(6), xticklabels=["P0", "P1", "P2", "P3", "P4", "P5"], title="启动阶段判定（各装置自身功率网格）")
    for row, device in enumerate(devices):
        for column, case in enumerate(device["cases"]):
            axes[0].text(column, row, f"{case['power_W']/1e3:g}", ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(left, ax=axes[0], ticks=[0, 1, 2, 3, 4], shrink=0.85)
    cbar.ax.set_yticklabels(["未越过阈值", "瞬时越过", "持续击穿", "预电离", "CFS候选"])

    mechanism_colors = ["#d9d9d9", "#6c757d", "#4361ee", "#3a86ff", "#f4a261", "#9b5de5", "#ffffff"]
    right = axes[1].imshow(mechanisms, aspect="auto", cmap=matplotlib.colors.ListedColormap(mechanism_colors), vmin=-0.5, vmax=len(mechanism_order)-0.5)
    axes[1].set(yticks=np.arange(len(devices)), yticklabels=[d["name"] for d in devices], xticks=np.arange(3), xticklabels=PHASE_LABELS, title="最高功率点主导电流机制")
    for row, device in enumerate(devices):
        for column, name in enumerate(device["cases"][-1]["dominant_mechanisms"]):
            axes[1].text(column, row, MECHANISM_CN[name], ha="center", va="center", fontsize=7)
    fig.colorbar(right, ax=axes[1], ticks=[], shrink=0.85)
    output = FIG_DIR / "startup_outcomes_and_mechanisms.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("titlecn", parent=base["Title"], fontName="ReportCN", fontSize=20, leading=27, textColor=colors.HexColor("#183b56"), spaceAfter=5 * mm),
        "h1": ParagraphStyle("h1cn", parent=base["Heading1"], fontName="ReportCN", fontSize=14, leading=19, textColor=colors.HexColor("#183b56"), spaceAfter=3 * mm),
        "h2": ParagraphStyle("h2cn", parent=base["Heading2"], fontName="ReportCN", fontSize=11, leading=15, textColor=colors.HexColor("#2a596f"), spaceBefore=2 * mm, spaceAfter=2 * mm),
        "body": ParagraphStyle("bodycn", parent=base["BodyText"], fontName="ReportCN", fontSize=9.0, leading=14, textColor=colors.HexColor("#222b31"), spaceAfter=2.5 * mm),
        "small": ParagraphStyle("smallcn", parent=base["BodyText"], fontName="ReportCN", fontSize=7.6, leading=11, textColor=colors.HexColor("#52606a"), spaceAfter=1.5 * mm),
        "warn": ParagraphStyle("warncn", parent=base["BodyText"], fontName="ReportCN", fontSize=9.0, leading=14, textColor=colors.HexColor("#8d241f"), backColor=colors.HexColor("#fff1ef"), borderPadding=6, spaceAfter=3 * mm),
    }


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("ReportCN", 7.5)
    canvas.setFillColor(colors.HexColor("#68747c"))
    canvas.drawString(16 * mm, 9 * mm, "linEC cross-device pre-CFS diagnostic study")
    canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def _table(data, widths, font_size=7.4, header=True) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "ReportCN"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c4ca")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8ee")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324d")),
            ]
        )
    table.setStyle(TableStyle(style))
    return table


def _fmt_current(value_A: float) -> str:
    return f"{value_A/1e3:.3g} kA" if abs(value_A) >= 1.0e3 else f"{value_A:.3g} A"


def build_pdf(devices: list[dict[str, Any]], figures: dict[str, Path]) -> None:
    styles = _styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
        title="Cross-device EC startup diagnostic study",
        author="linEC",
    )
    story: list[Any] = [
        Paragraph("不同托卡马克装置 EC 启动对比计算", styles["title"]),
        Paragraph("球形环优先：MAST、SUNIST、LATE、QUEST、VEST；常规托卡马克对照：DIII-D、J-TEXT。每台装置从独立、带注释的 INI 文件读取参数，并在各自配置的启动时间窗内使用同一套全局方程扫描 EC 功率。", styles["body"]),
        Paragraph("重要边界：这些输入不是逐炮完整重建，当前反应率、吸收、开场线损失和热尾电流仍含降阶闭合。报告中的 Ip 是 surrogate；在真实线圈波形、功率归一化 Dql、壁/限制器、气压/Zeff 和成功/失败炮验证门通过前，不能称为装置级准确预测。启动前没有使用 GS/EFIT 闭合磁面。", styles["warn"]),
    ]
    overview = [["装置", "类别", "A", "Bt", "EC", "位形", "CS口径"]]
    for device in devices:
        g, rf = device["geometry"], device["rf"]
        overview.append([
            device["name"],
            "球形/低A" if device["category"] == "spherical" else "常规",
            f"{g['aspect_ratio']:.2f}",
            f"{g['B_t_T']:.3g} T",
            f"{rf['frequency_Hz']/1e9:g} GHz n={rf['harmonic']}",
            g["field_configuration"],
            rf["cs_mode"],
        ])
    story.extend([
        _table(overview, [21*mm, 23*mm, 13*mm, 20*mm, 33*mm, 21*mm, 34*mm], 7.3),
        Spacer(1, 4*mm),
        Paragraph("如何重跑", styles["h1"]),
        Paragraph("修改 configs/devices 下相应 INI 的数值和带单引号的字符参数，再运行：python tools/build_cross_device_startup_report.py。每台装置的功率点来自该文件 [threshold_scan] 的 ec_powers_W。", styles["body"]),
        PageBreak(),
        Paragraph("1. 几何、环径比与谐波层", styles["h1"]),
        Image(str(figures["geometry"]), width=178*mm, height=69*mm),
        Paragraph("谐波层仅由真空环向场 B(R)=B0R0/R 标出，用来检查几何可达性；它不是吸收位置，更不是闭合磁面。O-X-B、热 EBW 吸收、折射和 Doppler/相对论展宽必须由逐时刻全波/射线+Dql 给出。", styles["body"]),
    ])
    geometry_rows = [["装置", "R0/a [m]", "p [mPa]", "Lconn [m]", "Rres [m]", "几何可达"]]
    for device in devices:
        g, r = device["geometry"], device["resonance"]
        geometry_rows.append([
            device["name"], f"{g['R0_m']:.3g}/{g['minor_radius_m']:.3g}", f"{device['gas_pressure_Pa']*1e3:.3g}", f"{g['connection_length_m']:.3g}", f"{r['resonance_R_m']:.3f}", "是" if r["resonance_in_geometric_cross_section"] else "否",
        ])
    story.extend([
        _table(geometry_rows, [27*mm, 31*mm, 25*mm, 28*mm, 28*mm, 28*mm], 7.6),
        PageBreak(),
        Paragraph("2. 球形环/低纵横比装置电流演化", styles["h1"]),
        Image(str(figures["spherical_traces"]), width=178*mm, height=162*mm),
        Paragraph("VEST 是 TPC + 低环压辅助案例，因此零 EC 功率仍可能出现感应电流；其结果不能与 MAST 的无螺线管 EBW 启动直接按 A/W 对比。其余四台装置在本矩阵中按无 CS 计算。", styles["body"]),
        PageBreak(),
        Paragraph("3. 常规托卡马克对照", styles["h1"]),
        Image(str(figures["conventional_traces"]), width=178*mm, height=57*mm),
        Paragraph("DIII-D 表示低电压欧姆 + ECH 辅助启动，EC 的主要作用可以是提高电离、电导和降低环压需求；J-TEXT 表示无 CS 的 105 GHz X2 假设案例。两者物理问题不同，所以报告同时给总峰值和相对零功率基线的 EC 增量。", styles["body"]),
        PageBreak(),
        Paragraph("4. 不同功率下可达到的电流", styles["h1"]),
        Image(str(figures["power"]), width=178*mm, height=100*mm),
        Paragraph("MAST 星号是文献锚点，不参与拟合。统一 0D 模型显著低估 100 kW MAST 电流，说明缺失经验证的 EBW 波粒动量输入与垂直场斜坡；这个差值不能通过修改一个隐藏 A/W 常数消除。", styles["warn"]),
        PageBreak(),
        Paragraph("5. 阶段判定与主导 CD 机制", styles["h1"]),
        Image(str(figures["outcomes"]), width=178*mm, height=80*mm),
        Paragraph("左图每格数字为 kW，颜色来自持续事件判据；右图把每台装置自己的配置时间窗按前20%、中间30%、后50%分段，再按最高功率点的电流分项绝对均值判定主导机制。TPC/VFC 仍是磁场位形，不被当成电流机制。", styles["body"]),
    ])

    outcome_cn = {
        "no sustained breakdown": "未越过击穿阈值",
        "transient breakdown-density crossing": "瞬时越过击穿密度",
        "breakdown": "持续击穿",
        "preionization": "预电离",
        "CFS candidate": "CFS 候选",
    }
    result_rows = [["装置", "Pmax", "peak Ip", "EC增量", "peak ne", "peak Te", "最高阶段"]]
    for device in devices:
        case = device["cases"][-1]
        result_rows.append([
            device["name"], f"{case['power_W']/1e3:g} kW", _fmt_current(case["peak_Ip_A"]), _fmt_current(case["ec_attributable_peak_increment_A"]), f"{case['peak_ne_m3']:.2e}", f"{case['peak_Te_eV']:.3g} eV", Paragraph(outcome_cn[case["outcome"]], styles["small"]),
        ])
    story.extend([
        _table(result_rows, [22*mm, 22*mm, 25*mm, 24*mm, 31*mm, 24*mm, 27*mm], 7.1),
        Paragraph("VEST 的零功率点在低环压下满足持续击穿，而较高 EC 功率点只瞬时越过密度阈值。用 0.05 ms 采样复核后，高功率点同时满足 ne 阈值与正净增长的持续时间仍只有约 0.05 ms；这是当前中性耗尽/开场线损失闭合产生的非单调结果，需要气压、壁源和 Halpha 时序验证。", styles["small"]),
        PageBreak(),
        Paragraph("6. 不能启动时如何解释", styles["h1"]),
    ])
    for device in devices:
        case = device["cases"][-1]
        mechanisms = " / ".join(MECHANISM_CN[name] for name in case["dominant_mechanisms"])
        story.extend([
            Paragraph(f"{device['name']} - {case['power_W']/1e3:g} kW", styles["h2"]),
            Paragraph(f"结果：peak Ip={_fmt_current(case['peak_Ip_A'])}，peak ne={case['peak_ne_m3']:.2e} m^-3，peak Te={case['peak_Te_eV']:.3g} eV；分段机制={mechanisms}。{device['highest_power_failure_explanation']}", styles["body"]),
        ])
    story.extend([
        Paragraph("这里的失败解释是模型内部的可检验归因，不是对实验的最终裁决。特别是 MAST/QUEST 的 EBW、VEST 的真实 TPC 场线和 J-TEXT 的实际端口/气压若未进入 Dql 与轨道损失，模型可能因缺物理而失败。", styles["warn"]),
        PageBreak(),
        Paragraph("7. 文献锚点、输入来源与证据边界", styles["h1"]),
    ])
    evidence_rows = [["装置", "实验口径", "可直接使用的证据", "DOI"]]
    for device in devices:
        evidence_rows.append([
            Paragraph(device["name"], styles["small"]),
            Paragraph(device["experiment_class"], styles["small"]),
            Paragraph(device["evidence"], styles["small"]),
            Paragraph(device["doi"], styles["small"]),
        ])
    story.extend([
        _table(evidence_rows, [20*mm, 42*mm, 74*mm, 41*mm], 6.6),
        Spacer(1, 4*mm),
        Paragraph("直接文献约束主要覆盖装置尺寸、频率、实验类型和少数电流/功率锚点。当前 INI 中的气压、Lconn、吸收率、Zeff、简化垂直场和部分功率网格均在文件头注释为 representative assumption；它们是可修改的研究假设，不是被引用论文逐项支持的测量值。", styles["body"]),
        Paragraph("定量质量门", styles["h1"]),
        Paragraph("逐炮 PF/TF/CS/TPC/VFC 波形；真实第一壁与场线命中；实测气压和 Zeff/杂质；端口、极化、束腰和净注入功率；空间及速度空间归一化 Dql；成功/失败炮的 Ip、ne、Halpha/Dalpha、HXR/ECE、loop voltage 联合验证。只有全部通过后，才能把 surrogate Ip 升级为装置级启动预测。", styles["body"]),
        Paragraph("可复现输出", styles["h1"]),
        Paragraph("完整逐功率时间序列、阶段、无量纲量和解释保存在 runs/cross_device_startup/results.json；每台装置的解析后 SI 输入保存在 runs/cross_device_startup/resolved_inputs。", styles["body"]),
    ])
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> None:
    setup()
    devices = run_matrix()
    figures = {
        "geometry": plot_geometry(devices),
        "spherical_traces": plot_traces(devices, "spherical", "spherical_device_current_traces.png"),
        "conventional_traces": plot_traces(devices, "conventional", "conventional_device_current_traces.png"),
        "power": plot_power_response(devices),
        "outcomes": plot_outcomes(devices),
    }
    build_pdf(devices, figures)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
