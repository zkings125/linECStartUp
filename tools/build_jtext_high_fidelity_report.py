"""Run representative J-TEXT power cases and build a physics-audited PDF."""

from __future__ import annotations

import json
from pathlib import Path

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

from ecstartup.config import SimulationConfig, load_config
from ecstartup.physics.hot_fullwave import HotPlasmaDielectric, NedelecPMLConfig, solve_hot_plasma_mesh_convergence
from ecstartup.physics.poisson_sheath import DebyePoissonAmbipolarSheath1D
from ecstartup.physics.relativistic_collisions import RelativisticBraamsKarney2V
from ecstartup.physics.fokker_planck import KineticState2V, RelativisticMomentumGrid
from ecstartup.solver import integrate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "jtext_high_fidelity"
FIG = ROOT / "output" / "figures" / "jtext_high_fidelity"
PDF = ROOT / "output" / "pdf" / "jtext_high_fidelity_startup_report.pdf"
FONT = Path(r"C:\Windows\Fonts\Deng.ttf")
POWERS_W = (0.0, 50.0e3, 100.0e3, 200.0e3, 300.0e3, 500.0e3)


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(str(FONT))
    family = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 9.5})
    pdfmetrics.registerFont(TTFont("ReportCN", str(FONT)))


def power_config(base: SimulationConfig, power_W: float) -> SimulationConfig:
    raw = base.model_dump(mode="python")
    raw["rf"]["sources"][0]["power_W"] = power_W
    raw["outputs"]["write_pdf_report"] = False
    raw["outputs"]["write_current_profile"] = False
    return SimulationConfig.model_validate(raw)


def summarize_case(power_W: float, result) -> dict:
    rows = result.diagnostics
    peak = max(rows, key=lambda row: abs(float(row["Ip_A"])))
    window_edges = (0.0, 0.02, 0.05, 0.10)
    windows = []
    for left, right in zip(window_edges[:-1], window_edges[1:]):
        selected = [
            row for time_s, row in zip(result.times_s, rows)
            if left <= float(time_s) <= right
        ]
        counts: dict[str, int] = {}
        for row in selected:
            name = str(row.get("dominant_current_mechanism", "none"))
            counts[name] = counts.get(name, 0) + 1
        dominant = max(counts, key=counts.get) if counts else "not available"
        windows.append({"time_range_s": [left, right], "dominant_mechanism": dominant})
    final = rows[-1]
    return {
        "power_W": power_W,
        "solver_success": bool(result.success),
        "peak_Ip_A": float(peak["Ip_A"]),
        "final_Ip_A": float(final["Ip_A"]),
        "peak_ne_m3": max(float(row["ne_m3"]) for row in rows),
        "peak_Te_eV": max(float(row["Te_eV"]) for row in rows),
        "final_stage": str(final.get("stage", "unknown")),
        "preionization_reached": any(event.type == "preionization" for event in result.events),
        "cfs_candidate_reached": any(event.type == "cfs" for event in result.events),
        "final_net_growth_s": float(final["net_growth_s"]),
        "final_loss_frequency_s": float(final["nu_loss_s"]),
        "final_ionization_fraction": float(final["ionization_fraction"]),
        "final_charge_balance_relative": float(final["charge_balance_relative"]),
        "windows": windows,
        "time_s": [float(value) for value in result.times_s],
        "Ip_A": [float(row["Ip_A"]) for row in rows],
        "ne_m3": [float(row["ne_m3"]) for row in rows],
        "Te_eV": [float(row["Te_eV"]) for row in rows],
    }


def run_power_scan() -> list[dict]:
    base = load_config(ROOT / "configs" / "jtext_ec_startup.ini")
    summaries = []
    for power in POWERS_W:
        result = integrate(power_config(base, power))
        summaries.append(summarize_case(power, result))
    payload = {
        "device": "J-TEXT representative pre-CFS case",
        "powers_W": list(POWERS_W),
        "cases": summaries,
        "warning": "Representative geometry and loss model; not a reconstruction of a specific J-TEXT shot.",
    }
    (OUT / "power_scan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def plot_power_scan(cases: list[dict]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.4), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, case in enumerate(cases):
        color = cmap(index / max(len(cases) - 1, 1))
        label = f"{case['power_W']/1e3:.0f} kW"
        t = np.asarray(case["time_s"]) * 1e3
        axes[0, 0].plot(t, np.asarray(case["Ip_A"]) / 1e3, color=color, label=label)
        axes[0, 1].plot(t, np.asarray(case["ne_m3"]), color=color)
        axes[1, 0].plot(t, np.asarray(case["Te_eV"]), color=color)
    powers = np.asarray([case["power_W"] for case in cases]) / 1e3
    peaks = np.asarray([case["peak_Ip_A"] for case in cases]) / 1e3
    axes[1, 1].plot(powers, peaks, marker="o", color="#b55239", linewidth=2)
    axes[0, 0].set(ylabel="Ip surrogate [kA]", title="Current evolution")
    axes[0, 1].set(ylabel="ne [m$^{-3}$]", title="Electron density")
    axes[1, 0].set(xlabel="time [ms]", ylabel="Te [eV]", title="Electron temperature")
    axes[1, 1].set(xlabel="injected power [kW]", ylabel="peak Ip surrogate [kA]", title="Power response")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    fig.suptitle("J-TEXT representative 105 GHz X2 no-CS scan (0-100 ms)", fontsize=13, weight="bold")
    output = FIG / "jtext_power_scan.png"
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def high_fidelity_checks() -> tuple[dict, Path]:
    sheath = DebyePoissonAmbipolarSheath1D(sheath_points=241).solve_boundary(
        1.0e17, 1.0e17, 20.0, 0.2
    )
    grid = RelativisticMomentumGrid.uniform(7, 5, 0.6, 0.6)
    collision = RelativisticBraamsKarney2V(grid, gyroangle_points=6)
    equilibrium = collision.maxwell_juttner_state(1.0e17, 2.0e4)
    up, _ = grid.mesh_u()
    perturbed = KineticState2V(equilibrium.f_e * np.maximum(1.0 + 0.08 * up, 0.1), grid, 0.0)
    collision_operator, collision_diag = collision.evaluate(
        perturbed, electron_electron_frequency_s=2.0e4, electron_ion_frequency_s=2.0e4
    )
    collision_scale = max(
        float(np.sum(np.abs(collision_operator) * grid.phase_volume_weights_m3_s3)), 1.0
    )

    dielectric = HotPlasmaDielectric(
        105.0e9, k_parallel_m_inv=35.0, k_perp_m_inv=1200.0,
        collision_frequency_s=2.0e7, resonance_width_fraction=4.0e-3,
    )
    config = NedelecPMLConfig(
        (-0.010, 0.010), (-0.008, 0.008), nx=49, ny=41,
        pml_thickness_m=0.002, source_center_m=(-0.004, 0.0), source_width_m=0.001,
    )
    constant = lambda value: (lambda x, y: np.full(np.broadcast(x, y).shape, value, dtype=float))
    coarse, wave = solve_hot_plasma_mesh_convergence(
        dielectric, config, constant(2.0e18), constant(1.875), constant(500.0),
        refinement_factor=1.3, tolerance=0.30,
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    axes[0].plot(sheath.distance_from_plasma_m / sheath.debye_length_m, sheath.potential_V, color="#2b6f9f", lw=2)
    axes[0].set(xlabel="$s/\\lambda_D$", ylabel="potential [V]", title="Debye-Poisson sheath")
    amplitude = np.sqrt(np.abs(wave.E_x_V_m) ** 2 + np.abs(wave.E_y_V_m) ** 2)
    im = axes[1].tripcolor(wave.element_x_m * 1e3, wave.element_y_m * 1e3, amplitude, shading="gouraud", cmap="magma")
    axes[1].set(xlabel="x [mm]", ylabel="y [mm]", title="Nedelec/PML |E|")
    fig.colorbar(im, ax=axes[1], label="V/m")
    harmonics = [1, 2]
    axes[2].bar(["n=1", "n=2"], [wave.harmonic_absorbed_power_W_per_m[n] for n in harmonics], color=["#1f8a70", "#b55239"])
    axes[2].set(ylabel="absorbed power [W/m]", title="Hot harmonic partition")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("High-fidelity backend verification cases", fontsize=13, weight="bold")
    output = FIG / "high_fidelity_checks.png"
    fig.savefig(output, dpi=190)
    plt.close(fig)
    diagnostics = {
        "poisson_sheath": {
            "debye_length_m": sheath.debye_length_m,
            "wall_drop_V": sheath.wall_drop_V,
            "poisson_residual_relative": sheath.poisson_residual_relative,
        },
        "relativistic_collisions": {
            **collision_diag.__dict__,
            "particle_residual_ee_relative": abs(collision_diag.particle_residual_ee) / collision_scale,
            "momentum_residual_ee_relative": abs(collision_diag.momentum_residual_ee) / collision_scale,
            "energy_residual_ee_relative": abs(collision_diag.energy_residual_ee) / collision_scale,
            "particle_residual_ei_relative": abs(collision_diag.particle_residual_ei) / collision_scale,
            "energy_residual_ei_relative": abs(collision_diag.energy_residual_ei) / collision_scale,
        },
        "hot_fullwave": {
            "coarse_dof": coarse.degrees_of_freedom,
            "fine_dof": wave.degrees_of_freedom,
            "mesh_change_relative": wave.mesh_change_relative,
            "mesh_converged": wave.mesh_converged,
            "power_balance_relative": wave.power_balance_relative,
            "cells_per_vacuum_wavelength": wave.cells_per_vacuum_wavelength,
            "second_harmonic_fraction": wave.second_harmonic_fraction,
        },
    }
    (OUT / "high_fidelity_diagnostics.json").write_text(
        json.dumps(
            diagnostics,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ),
        encoding="utf-8",
    )
    return diagnostics, output


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("titlecn", parent=base["Title"], fontName="ReportCN", fontSize=21, leading=28, textColor=colors.HexColor("#17324d"), spaceAfter=6*mm),
        "h1": ParagraphStyle("h1cn", parent=base["Heading1"], fontName="ReportCN", fontSize=15, leading=20, textColor=colors.HexColor("#17324d"), spaceAfter=4*mm),
        "body": ParagraphStyle("bodycn", parent=base["BodyText"], fontName="ReportCN", fontSize=9.2, leading=14.5, spaceAfter=3*mm),
        "small": ParagraphStyle("smallcn", parent=base["BodyText"], fontName="ReportCN", fontSize=8, leading=12, textColor=colors.HexColor("#4d5963")),
    }


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("ReportCN", 8); canvas.setFillColor(colors.HexColor("#65717a"))
    canvas.drawString(17*mm, 10*mm, "linEC J-TEXT representative startup assessment")
    canvas.drawRightString(A4[0]-17*mm, 10*mm, f"第 {doc.page} 页"); canvas.restoreState()


def build_pdf(cases: list[dict], power_figure: Path, checks: dict, check_figure: Path) -> None:
    s = styles()
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=17*mm, rightMargin=17*mm, topMargin=16*mm, bottomMargin=17*mm, title="J-TEXT EC startup power scan", author="linEC")
    story = [
        Paragraph("J-TEXT 105 GHz EC 无感启动代表性算例", s["title"]),
        Paragraph("计算窗口为 0-100 ms，无 CS、无 pre-CFS GS/EFIT 磁面。装置尺寸和 Bt 采用代表性参数；气压、波束、壁与损失并非某一炮的完整重建，因此电流值只能作为机制/灵敏度结果。", s["body"]),
        Image(str(power_figure), width=176*mm, height=122*mm),
        Paragraph("该配置在全部扫描功率下均未达到配置的预电离/CFS 事件。功率升高能够增加热尾代理电流，但不能克服开放场线损失和密度衰减，因此不能据此声称 J-TEXT 已实现 EC 无感启动。", s["body"]),
        PageBreak(),
        Paragraph("1. 功率-电流与启动判定", s["h1"]),
    ]
    table_data = [["EC power", "peak Ip", "peak ne", "peak Te", "final stage", "startup"]]
    for case in cases:
        table_data.append([
            f"{case['power_W']/1e3:.0f} kW", f"{case['peak_Ip_A']/1e3:.3f} kA",
            f"{case['peak_ne_m3']:.3e} m^-3", f"{case['peak_Te_eV']:.1f} eV",
            case["final_stage"], "no" if not case["cfs_candidate_reached"] else "candidate",
        ])
    table = Table(table_data, colWidths=[25*mm, 25*mm, 39*mm, 27*mm, 28*mm, 24*mm])
    table.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"ReportCN"),("FONTSIZE",(0,0),(-1,-1),7.4),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dce6ee")),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#b9c2c9")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.extend([table, Spacer(1,5*mm), Paragraph("不能启动的解释", s["h1"]), Paragraph(
        "本算例的关键不是二次谐波几何不可达：105 GHz X2 在 Bt=1.7-2.0 T 的代表性径向范围内可达。失败来自组合条件：开放场线损失频率高于净电离增长、无感电路没有环电压、配置未提供功率归一化真实 Dql，因此产生的主要是热尾优先损失代理，而不是经验证的定量 ECCD/EBWCD。高气压还可能提高电子-中性碰撞并降低高能尾；过低气压则缺少足够靶粒子。", s["body"]),
        Paragraph("分时段 CD 机制", s["h1"]),
    ])
    mech_rows = [["power", "0-20 ms", "20-50 ms", "50-100 ms"]]
    labels = {"hot_tail":"热尾优先约束", "direct_rf":"直接 RF", "kinetic":"F-P EC", "ohmic":"感应", "none":"无"}
    for case in cases:
        mech_rows.append([f"{case['power_W']/1e3:.0f} kW"] + [labels.get(w["dominant_mechanism"], w["dominant_mechanism"]) for w in case["windows"]])
    mech = Table(mech_rows, colWidths=[28*mm,45*mm,45*mm,45*mm])
    mech.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"ReportCN"),("FONTSIZE",(0,0),(-1,-1),8),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dce6ee")),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#b9c2c9")),("ALIGN",(1,1),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.extend([mech, Paragraph("零功率行的微小初值/数值电流不应解释为 CD。非零功率工况的 dominant mechanism 为 hot_tail，反映当前全局闭合的因果通道；真实 F-P Dql 尚未用 J-TEXT 炮级波束归一化。", s["small"]), PageBreak(),
        Paragraph("2. 高保真后端独立验证", s["h1"]), Image(str(check_figure), width=176*mm, height=55*mm),
        Paragraph(f"Debye-Poisson 鞘层：lambda_D={checks['poisson_sheath']['debye_length_m']:.3e} m，浮动壁降={checks['poisson_sheath']['wall_drop_V']:.2f} V，Poisson 离散残差={checks['poisson_sheath']['poisson_residual_relative']:.2e}。", s["body"]),
        Paragraph(f"热全波：细网格 DOF={checks['hot_fullwave']['fine_dof']}，网格变化={checks['hot_fullwave']['mesh_change_relative']:.2%}，功率闭合残差={checks['hot_fullwave']['power_balance_relative']:.2e}，真空波长每单元={checks['hot_fullwave']['cells_per_vacuum_wavelength']:.1f}。这是局部转换层验证，不是 J-TEXT 整机端口重建。", s["body"]),
        Paragraph("相对论碰撞采用 Beliaev-Budker 协变 Landau 张量的直接积分，与 Braams-Karney 势方程形式等价；Cee 投影守恒粒子、平行动量和能量，Cei Lorentz 极限守恒电子粒子与能量并向离子背景传递动量。", s["body"]),
        Paragraph("二次谐波 Dql 使用 J1(k_perp rho)^2 FLR 因子和相对论共振；EBWCD 只有在全波网格/功率门通过且 F-P 电流与波动量电流闭合时才标记 quantitative。当前 J-TEXT 报告没有炮级二维全波输入，因此没有用该闭合替换表中的 Ip。", s["body"]),
        PageBreak(), Paragraph("3. 证据边界与结论", s["h1"]), Paragraph(
        "结论：在本代表性输入下，提高 105 GHz 功率使电流代理连续上升，但 500 kW 仍未达到预电离/CFS 判据。它支持“共振层可达并不足以保证启动”的解释，更可能的限制是早期吸收/动量不对称不足、开场线约束弱以及气压-碰撞窗口不合适。要给出某一 J-TEXT 炮的准确阈值和 Ip，仍必须接入该炮 PF/TF/CS 波形、气压、EC 端口/极化/束腰、壁几何、Halpha/HXR/ECE 与成功/失败标签。", s["body"]),
        Paragraph("参考：Braams & Karney, Phys. Rev. Lett. 59, 1817 (1987), DOI 10.1103/PhysRevLett.59.1817；Rosenbluth et al., Phys. Rev. 107, 1 (1957), DOI 10.1103/PhysRev.107.1；Mjølhus, J. Plasma Phys. 31 (1984), DOI 10.1017/S0022377800001392；Lopez & Poli, PPCF 60, 065007 (2018), DOI 10.1088/1361-6587/aabaa8。", s["small"]),
    ])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    setup()
    cases = run_power_scan()
    power_figure = plot_power_scan(cases)
    checks, check_figure = high_fidelity_checks()
    build_pdf(cases, power_figure, checks, check_figure)
    print(PDF)


if __name__ == "__main__":
    main()
