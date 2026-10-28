"""Calibrate temporary MAST EBW momentum and VFC-ramp inputs to literature anchors."""

from __future__ import annotations

import argparse
import ast
import configparser
from dataclasses import dataclass
import csv
import json
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
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.optimize import brentq

from ecstartup.config import SimulationConfig, load_config
from ecstartup.physics.calibrated_ebw_momentum import (
    CalibratedEBWMomentumDrive,
    CalibratedEBWSettings,
)
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.solver import RunResult, integrate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "calibration" / "mast_ebw_anchor.ini"
FONT_PATH = Path(r"C:\Windows\Fonts\Deng.ttf")


@dataclass(frozen=True)
class AnchorInput:
    base_config: Path
    device_name: str
    evidence_class: str
    reference_doi: str
    injected_power_W: float
    ordinary_target_A: float
    optimized_target_A: float
    power_scan_W: tuple[float, ...]
    eta_bounds: tuple[float, float]
    waveform_T: tuple[tuple[float, float], ...]
    ramp_shape_exponent: float
    gain_bounds: tuple[float, float]
    run_directory: Path
    figure_directory: Path
    pdf_file: Path
    write_pdf: bool


@dataclass
class Case:
    name: str
    power_W: float
    ramp_enabled: bool
    eta_A_per_W: float
    ramp_gain_max: float
    config: SimulationConfig
    settings: CalibratedEBWSettings
    result: RunResult

    @property
    def peak_current_A(self) -> float:
        return max(float(row["Ip_A"]) for row in self.result.diagnostics)


def _quoted(value: str, name: str) -> str:
    text = value.strip()
    if len(text) < 2 or text[0] != "'" or text[-1] != "'":
        raise ValueError(f"{name} must be enclosed in single quotes")
    return text[1:-1]


def _literal(value: str, name: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid literal for {name}: {value}") from exc


def _path(value: str, name: str) -> Path:
    path = Path(_quoted(value, name))
    return path if path.is_absolute() else ROOT / path


def load_anchor_input(path: Path) -> AnchorInput:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(path)
    required = {"calibration", "ebw_momentum", "vertical_field_ramp", "outputs"}
    if set(parser.sections()) != required:
        raise ValueError(f"calibration INI sections must be exactly {sorted(required)}")
    cal = parser["calibration"]
    ebw = parser["ebw_momentum"]
    ramp = parser["vertical_field_ramp"]
    out = parser["outputs"]
    if _quoted(ebw["switchOn"], "ebw_momentum.switchOn") != "enabled":
        raise ValueError("EBW momentum calibration is disabled")
    if _quoted(ramp["switchOn"], "vertical_field_ramp.switchOn") != "enabled":
        raise ValueError("vertical-field ramp calibration is disabled")
    if _quoted(ebw["normalization"], "ebw_momentum.normalization") != "absorbed_power":
        raise ValueError("only absorbed_power normalization is supported")
    powers = tuple(float(item) for item in _literal(cal["power_scan_W"], "power_scan_W"))
    waveform = tuple(
        (float(time), float(field))
        for time, field in _literal(ramp["waveform_T"], "waveform_T")
    )
    if len(waveform) < 2:
        raise ValueError("vertical-field waveform requires at least two points")
    write_pdf = _quoted(out["write_pdf_report"], "outputs.write_pdf_report") == "enabled"
    return AnchorInput(
        base_config=_path(cal["base_config"], "calibration.base_config"),
        device_name=_quoted(cal["device_name"], "calibration.device_name"),
        evidence_class=_quoted(cal["evidence_class"], "calibration.evidence_class"),
        reference_doi=_quoted(cal["reference_doi"], "calibration.reference_doi"),
        injected_power_W=float(cal["injected_power_W"]),
        ordinary_target_A=float(cal["ordinary_field_target_current_A"]),
        optimized_target_A=float(cal["optimized_ramp_target_current_A"]),
        power_scan_W=powers,
        eta_bounds=(float(ebw["efficiency_lower_A_per_W"]), float(ebw["efficiency_upper_A_per_W"])),
        waveform_T=waveform,
        ramp_shape_exponent=float(ramp["ramp_shape_exponent"]),
        gain_bounds=(float(ramp["gain_lower"]), float(ramp["gain_upper"])),
        run_directory=_path(out["run_directory"], "outputs.run_directory"),
        figure_directory=_path(out["figure_directory"], "outputs.figure_directory"),
        pdf_file=_path(out["pdf_file"], "outputs.pdf_file"),
        write_pdf=write_pdf,
    )


def _set_power(config: SimulationConfig, power_W: float) -> SimulationConfig:
    sources = []
    for source in config.rf.sources:
        sources.append(source.model_copy(update={"power_W": float(power_W)}))
    rf = config.rf.model_copy(update={"sources": sources})
    return config.model_copy(update={"rf": rf})


def _case_config(base: SimulationConfig, power_W: float, ramp: bool, waveform: tuple[tuple[float, float], ...]) -> SimulationConfig:
    config = _set_power(base, power_W)
    geometry = config.geometry.model_copy(
        update={"vertical_field_waveform": list(waveform) if ramp else None}
    )
    return config.model_copy(update={"geometry": geometry})


def run_case(
    base: SimulationConfig,
    spec: AnchorInput,
    *,
    name: str,
    power_W: float,
    eta_A_per_W: float,
    ramp_gain_max: float,
    ramp_enabled: bool,
) -> Case:
    config = _case_config(base, power_W, ramp_enabled, spec.waveform_T)
    settings = CalibratedEBWSettings(
        current_per_absorbed_power_A_per_W=eta_A_per_W,
        ramp_gain_max=ramp_gain_max,
        ramp_shape_exponent=spec.ramp_shape_exponent,
        ramp_start_field_T=spec.waveform_T[0][1],
        ramp_end_field_T=spec.waveform_T[-1][1],
        vertical_field_waveform=spec.waveform_T,
        ramp_enabled=ramp_enabled,
    )
    drive = CalibratedEBWMomentumDrive(config.geometry, config.rf, settings)
    model = GlobalStartupModel(config, current_drive=drive)
    return Case(name, power_W, ramp_enabled, eta_A_per_W, ramp_gain_max, config, settings, integrate(config, model))


def _solve_parameter(function, bounds: tuple[float, float], target: float, label: str) -> float:
    lower, upper = bounds
    f_lower = function(lower) - target
    f_upper = function(upper) - target
    if f_lower == 0.0:
        return lower
    if f_upper == 0.0:
        return upper
    if f_lower * f_upper > 0.0:
        raise RuntimeError(
            f"{label} target is not bracketed: lower error={f_lower:.3e} A, upper error={f_upper:.3e} A"
        )
    return float(brentq(lambda value: function(value) - target, lower, upper, xtol=1.0e-7, rtol=1.0e-9))


def _series(case: Case) -> dict[str, np.ndarray]:
    rows = case.result.diagnostics
    t = np.asarray(case.result.times_s, dtype=float)
    absorbed = np.asarray([float(row["rf_power_absorbed_W"]) for row in rows])
    bv = np.asarray([float(row["B_vertical_T"]) for row in rows])
    base_current = case.eta_A_per_W * absorbed
    gain = np.asarray([case.settings.ramp_gain(value) for value in bv])
    return {
        "time_s": t,
        "Ip_A": np.asarray([float(row["Ip_A"]) for row in rows]),
        "I_ohmic_A": np.asarray([float(row["I_ohmic_A"]) for row in rows]),
        "I_ebw_A": np.asarray([float(row["I_rf_A"]) for row in rows]),
        "I_tail_A": np.asarray([float(row["I_tail_A"]) for row in rows]),
        "I_vertical_A": np.asarray([float(row["I_vertical_charge_A"]) for row in rows]),
        "absorbed_power_W": absorbed,
        "base_ebw_momentum_current_A": base_current,
        "ramp_gain": gain,
        "enhanced_ebw_momentum_current_A": base_current * gain,
        "B_vertical_T": bv,
        "dB_vertical_dt_T_per_s": np.asarray([case.settings.ramp_rate_T_per_s(value) for value in t]),
    }


def _mechanism_windows(case: Case) -> list[dict[str, Any]]:
    values = _series(case)
    windows = ((0.0, 0.020), (0.020, 0.050), (0.050, 0.100001))
    names = {
        "I_ohmic_A": "CS/inductive",
        "I_ebw_A": "calibrated EBW momentum hypothesis",
        "I_tail_A": "hot-tail preferential confinement",
        "I_vertical_A": "vertical charge separation",
    }
    output = []
    for left, right in windows:
        mask = (values["time_s"] >= left) & (values["time_s"] < right)
        means = {names[key]: float(np.mean(np.abs(values[key][mask]))) for key in names}
        dominant = max(means, key=means.get)
        output.append({"time_ms": [left * 1e3, min(right, 0.1) * 1e3], "dominant": dominant, "mean_abs_A": means})
    return output


def _save_generated_inputs(case: Case, run_dir: Path) -> None:
    values = _series(case)
    np.savez_compressed(run_dir / "generated_ebw_vfc_inputs.npz", **values)
    keys = [
        "time_s", "absorbed_power_W", "base_ebw_momentum_current_A", "ramp_gain",
        "enhanced_ebw_momentum_current_A", "B_vertical_T", "dB_vertical_dt_T_per_s",
    ]
    with (run_dir / "generated_ebw_vfc_inputs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(values[key] for key in keys)))


def _style_axes(axis) -> None:
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def _make_figures(cases: dict[str, Case], scans: list[tuple[Case, Case]], spec: AnchorInput) -> list[Path]:
    spec.figure_directory.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    font = None
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        font = font_manager.FontProperties(fname=str(FONT_PATH))
        plt.rcParams["font.family"] = font.get_name()
        plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for key, label, color in (
        ("no_ebw", "无 EBW 动量输入", "#777777"),
        ("ordinary", "EBW 动量假设校准，恒定 Bv", "#176B87"),
        ("optimized", "EBW 动量假设 + VFC 斜坡", "#C03D3E"),
    ):
        series = _series(cases[key])
        ax.plot(series["time_s"] * 1e3, series["Ip_A"] / 1e3, label=label, color=color, linewidth=2.0)
    ax.axhline(spec.ordinary_target_A / 1e3, color="#176B87", linestyle="--", alpha=0.65)
    ax.axhline(spec.optimized_target_A / 1e3, color="#C03D3E", linestyle="--", alpha=0.65)
    ax.set(xlabel="时间 [ms]", ylabel="Ip [kA]", title="MAST 文献锚点充分性测试")
    ax.legend(prop=font, frameon=False)
    _style_axes(ax)
    path = spec.figure_directory / "anchor_current_evolution.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); figures.append(path)

    series = _series(cases["optimized"])
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
    axes[0].plot(series["time_s"] * 1e3, series["B_vertical_T"] * 1e3, color="#176B87", linewidth=2)
    axes[0].set_ylabel("Bv [mT]")
    axes[1].plot(series["time_s"] * 1e3, series["dB_vertical_dt_T_per_s"], color="#5B7F3A", linewidth=2)
    axes[1].set_ylabel("dBv/dt [T/s]")
    axes[2].plot(series["time_s"] * 1e3, series["ramp_gain"], color="#C03D3E", linewidth=2)
    axes[2].set(xlabel="时间 [ms]", ylabel="动量保留增益")
    axes[0].set_title("生成的 VFC 斜坡与显式校准增益（非实测波形）")
    for axis in axes: _style_axes(axis)
    path = spec.figure_directory / "generated_vfc_ramp.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); figures.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.1), sharex=True)
    axes[0].plot(series["time_s"] * 1e3, series["absorbed_power_W"] / 1e3, color="#5B7F3A", linewidth=2)
    axes[0].set(ylabel="吸收功率 [kW]", title="生成的 EBW 波粒动量输入")
    axes[1].plot(series["time_s"] * 1e3, series["base_ebw_momentum_current_A"] / 1e3, label="基础 EBW 动量", color="#176B87")
    axes[1].plot(series["time_s"] * 1e3, series["enhanced_ebw_momentum_current_A"] / 1e3, label="含斜坡保留", color="#C03D3E", linewidth=2)
    axes[1].set(xlabel="时间 [ms]", ylabel="等效驱动电流 [kA]")
    axes[1].legend(prop=font, frameon=False)
    for axis in axes: _style_axes(axis)
    path = spec.figure_directory / "generated_ebw_momentum.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); figures.append(path)

    powers = np.asarray([ordinary.power_W for ordinary, _ in scans]) / 1e3
    ordinary_peak = np.asarray([ordinary.peak_current_A for ordinary, _ in scans]) / 1e3
    optimized_peak = np.asarray([optimized.peak_current_A for _, optimized in scans]) / 1e3
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.plot(powers, ordinary_peak, "o-", label="恒定 Bv", color="#176B87")
    ax.plot(powers, optimized_peak, "s-", label="生成 VFC 斜坡", color="#C03D3E")
    ax.scatter([100, 100], [17, 33], marker="*", s=150, color="#111111", label="文献锚点", zorder=5)
    ax.set(xlabel="注入功率 [kW]", ylabel="峰值 Ip [kA]", title="固定校准系数后的功率外推")
    ax.legend(prop=font, frameon=False)
    _style_axes(ax)
    path = spec.figure_directory / "power_scan.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); figures.append(path)

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for key, label, color in (
        ("I_ebw_A", "EBW 动量假设", "#C03D3E"),
        ("I_tail_A", "热尾", "#5B7F3A"),
        ("I_vertical_A", "垂直电荷分离", "#8F6CAB"),
        ("I_ohmic_A", "CS/感应", "#777777"),
    ):
        ax.plot(series["time_s"] * 1e3, np.abs(series[key]) / 1e3, label=label, color=color)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set(xlabel="时间 [ms]", ylabel="|电流分项| [kA]", title="优化案例的电流机制分解")
    ax.legend(prop=font, frameon=False)
    _style_axes(ax)
    path = spec.figure_directory / "mechanism_decomposition.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); figures.append(path)
    return figures


def _register_font() -> str:
    if FONT_PATH.exists():
        pdfmetrics.registerFont(TTFont("Deng", str(FONT_PATH)))
        return "Deng"
    return "Helvetica"


def _table(data, widths=None, font_name: str = "Helvetica"):
    cell_style = ParagraphStyle(
        "TableCell",
        fontName=font_name,
        fontSize=8.5,
        leading=11,
    )
    rendered = [
        [Paragraph(str(cell), cell_style) for cell in row]
        for row in data
    ]
    table = Table(rendered, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7EEF2")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#88949A")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _write_pdf(spec: AnchorInput, cases: dict[str, Case], scans: list[tuple[Case, Case]], figures: list[Path], eta: float, gain: float) -> None:
    spec.pdf_file.parent.mkdir(parents=True, exist_ok=True)
    font = _register_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("CNBody", parent=styles["BodyText"], fontName=font, fontSize=9.2, leading=14)
    title = ParagraphStyle("CNTitle", parent=styles["Title"], fontName=font, fontSize=18, leading=24, textColor=colors.HexColor("#163A4A"))
    head = ParagraphStyle("CNHead", parent=styles["Heading2"], fontName=font, fontSize=13, leading=18, textColor=colors.HexColor("#176B87"))
    small = ParagraphStyle("CNSmall", parent=body, fontSize=8, leading=11)
    doc = SimpleDocTemplate(str(spec.pdf_file), pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm)
    story = [
        Paragraph("MAST EBW 波粒动量与垂直场斜坡：文献锚点充分性测试", title),
        Spacer(1, 4*mm),
        Paragraph("结论：在当前全局模型中，显式加入按吸收功率归一化的 EBW 动量假设后可校准到 100 kW → 17 kA；保持同一基础效率，仅加入生成的 VFC 斜坡及其动量保留增益，可校准到约 33 kA。这个充分性测试不证明 O-X-B 真的发生，也不证明所生成波形或增益是真实 MAST 炮级输入。", body),
        Spacer(1, 3*mm),
        _table([
            ["项目", "数值", "证据状态"],
            ["参考", f"DOI {spec.reference_doi}", "文献锚点"],
            ["EBW 动量假设效率", f"{eta:.6f} A/W absorbed", "由 17 kA 反演；未确认模式转换"],
            ["VFC 最大保留增益", f"{gain:.6f}", "由 33 kA 反演"],
            ["生成 Bv 波形", "6 mT → 14 mT (20–60 ms)", "非实测、非炮级"],
            ["GS/EFIT", "0–100 ms 未使用", "符合 pre-CFS 限制"],
        ], [42*mm, 67*mm, 50*mm], font),
        Spacer(1, 4*mm), Image(str(figures[0]), width=174*mm, height=100*mm),
        PageBreak(), Paragraph("生成的两类输入", head),
        Paragraph("EBW 输入写成 I_EBW,base(t)=η_EBW·P_abs(t)。VFC 斜坡不直接作为电流源，而通过 G_ramp=1+(Gmax-1)·progress(Bv)^p 改变已吸收波动量的保留。原始动量、增益和增强后动量分别输出，避免用一个隐藏常数同时拟合两个锚点。", body),
        Image(str(figures[1]), width=174*mm, height=153*mm), PageBreak(),
        Image(str(figures[2]), width=174*mm, height=130*mm),
        Paragraph("机器可读输入：generated_ebw_vfc_inputs.csv / .npz。它们可以替换为未来获得的全波 Dql 动量矩和实测 VFC 线圈波形，而不改全局求解器。", small),
        PageBreak(), Paragraph("外推与机制", head), Image(str(figures[3]), width=174*mm, height=100*mm),
        Paragraph("功率扫描保持 η_EBW 和 Gmax 不变，因此除 100 kW 两个校准点外均是模型外推。它只能检验趋势，不能作为 MAST 或其他装置的启动阈值预测。", body),
        Spacer(1, 3*mm), Image(str(figures[4]), width=174*mm, height=100*mm),
    ]
    stages = _mechanism_windows(cases["optimized"])
    story.extend([PageBreak(), Paragraph("分时段主导机制", head)])
    stage_rows = [["时段 [ms]", "主导 CD 机制", "平均 |I| [kA]"]]
    for row in stages:
        dominant = row["dominant"]
        stage_rows.append([
            f"{row['time_ms'][0]:.0f}–{row['time_ms'][1]:.0f}",
            dominant,
            f"{row['mean_abs_A'][dominant]/1e3:.3f}",
        ])
    story.extend([
        _table(stage_rows, [35*mm, 80*mm, 45*mm], font), Spacer(1, 4*mm),
        Paragraph("边界与下一步：本报告没有恢复 MAST 原始 VFC 线圈电流、O-X-B 转换效率、EBW k_parallel 谱或全波 Dql；也没有把 ramp gain 解释成第一性原理项。正式验证需要用实测线圈波形替换 Bv(t)，并由全波/F-P 输出 ∫p_parallel QL[f] d³p 的动量矩，再检查功率与动量守恒以及独立炮号的 Ip(t)。", body),
    ])
    doc.build(story)


def calibrate(spec: AnchorInput) -> dict[str, Any]:
    base = load_config(spec.base_config)
    cache: dict[tuple, Case] = {}

    def cached(name: str, power: float, eta: float, gain: float, ramp: bool) -> Case:
        key = (round(power, 6), round(eta, 12), round(gain, 12), ramp)
        if key not in cache:
            cache[key] = run_case(base, spec, name=name, power_W=power, eta_A_per_W=eta, ramp_gain_max=gain, ramp_enabled=ramp)
        return cache[key]

    eta = _solve_parameter(
        lambda value: cached("eta_search", spec.injected_power_W, value, 1.0, False).peak_current_A,
        spec.eta_bounds,
        spec.ordinary_target_A,
        "EBW efficiency",
    )
    gain = _solve_parameter(
        lambda value: cached("gain_search", spec.injected_power_W, eta, value, True).peak_current_A,
        spec.gain_bounds,
        spec.optimized_target_A,
        "VFC retention gain",
    )
    cases = {
        "no_ebw": cached("no_ebw", spec.injected_power_W, 0.0, 1.0, False),
        "ordinary": cached("ordinary", spec.injected_power_W, eta, 1.0, False),
        "optimized": cached("optimized", spec.injected_power_W, eta, gain, True),
    }
    scans = [
        (
            cached(f"ordinary_{power:.0f}", power, eta, 1.0, False),
            cached(f"optimized_{power:.0f}", power, eta, gain, True),
        )
        for power in spec.power_scan_W
    ]
    spec.run_directory.mkdir(parents=True, exist_ok=True)
    _save_generated_inputs(cases["optimized"], spec.run_directory)
    figures = _make_figures(cases, scans, spec)
    summary = {
        "status": "literature_anchor_sufficiency_test_not_device_prediction",
        "device": spec.device_name,
        "reference_doi": spec.reference_doi,
        "calibrated_parameters": {
            "ebw_current_per_absorbed_power_A_per_W": eta,
            "vfc_ramp_retention_gain_max": gain,
            "vfc_waveform_T": spec.waveform_T,
            "vfc_waveform_provenance": "generated_not_shot_resolved",
        },
        "anchor_cases": {
            key: {
                "peak_current_A": case.peak_current_A,
                "target_current_A": spec.ordinary_target_A if key == "ordinary" else spec.optimized_target_A if key == "optimized" else None,
                "relative_error": (
                    abs(case.peak_current_A - (spec.ordinary_target_A if key == "ordinary" else spec.optimized_target_A))
                    / (spec.ordinary_target_A if key == "ordinary" else spec.optimized_target_A)
                    if key != "no_ebw" else None
                ),
            }
            for key, case in cases.items()
        },
        "power_scan": [
            {"power_W": ordinary.power_W, "ordinary_peak_A": ordinary.peak_current_A, "optimized_peak_A": optimized.peak_current_A}
            for ordinary, optimized in scans
        ],
        "optimized_stage_mechanisms": _mechanism_windows(cases["optimized"]),
        "generated_inputs": {
            "csv": str(spec.run_directory / "generated_ebw_vfc_inputs.csv"),
            "npz": str(spec.run_directory / "generated_ebw_vfc_inputs.npz"),
        },
        "limitations": [
            "EBW efficiency is inverse-calibrated from the 17 kA literature anchor.",
            "The VFC waveform is generated and the retention gain is inverse-calibrated from 33 kA.",
            "No pre-CFS GS/EFIT equilibrium is used.",
            "The result is not a quantitative device prediction or a validated O-X-B/EBW calculation.",
        ],
    }
    (spec.run_directory / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if spec.write_pdf:
        _write_pdf(spec, cases, scans, figures, eta, gain)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = calibrate(load_anchor_input(args.config.resolve()))
    print(json.dumps(summary["calibrated_parameters"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["anchor_cases"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
