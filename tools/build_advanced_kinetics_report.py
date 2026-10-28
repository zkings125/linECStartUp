"""Build figures and a visually checked PDF for the advanced kinetic examples."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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

from ecstartup.physics.fokker_planck import (
    QuasilinearDiffusion2V,
    RelativisticFokkerPlanck2V,
    RelativisticMomentumGrid,
)
from ecstartup.physics.orbit_average import classify_pitch
from ecstartup.physics.startup_resonance import resonance_field_T


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "output" / "figures" / "advanced_kinetics"
PDF_PATH = ROOT / "output" / "pdf" / "exl50u_advanced_kinetics_report.pdf"
JTEXT_JSON = ROOT / "runs" / "jtext_ec_startup" / "resonance_assessment.json"
EXL_DIR = ROOT / "runs" / "exl50u_formal"
FONT_PATH = Path(r"C:\Windows\Fonts\Deng.ttf")


def setup_fonts() -> None:
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 130,
        "savefig.dpi": 190,
    })
    pdfmetrics.registerFont(TTFont("Deng", str(FONT_PATH)))


def style_axis(ax) -> None:
    ax.grid(True, color="#d5d9dc", lw=0.55, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)


def plot_jtext_resonance() -> Path:
    assessment = json.loads(JTEXT_JSON.read_text(encoding="utf-8"))
    R0, a = 1.05, 0.255
    R = np.linspace(0.68, 1.42, 500)
    fields = [1.70, 1.85, 2.00]
    B1 = resonance_field_T(105.0e9, 1)
    B2 = resonance_field_T(105.0e9, 2)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.35), constrained_layout=True)
    ax = axes[0]
    ax.axvspan(R0 - a, R0 + a, color="#dcefe5", alpha=0.75, label="代表性等离子体径向范围")
    for Bt, color in zip(fields, ("#2b6f9f", "#1f8a70", "#b55239")):
        ax.plot(R, Bt * R0 / R, lw=2.0, color=color, label=fr"$B_t(R_0)={Bt:.2f}$ T")
    ax.axhline(B2, color="#8d5bb5", lw=1.8, ls="--", label=f"105 GHz X2: {B2:.3f} T")
    ax.axhline(B1, color="#343a40", lw=1.5, ls=":", label=f"105 GHz X1: {B1:.3f} T")
    ax.set(xlabel="大半径 R [m]", ylabel="真空环向场 B(R) [T]", title="J-TEXT 105 GHz 共振层可达性")
    ax.set_ylim(1.15, 4.25)
    style_axis(ax)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    Bt_scan = np.linspace(1.65, 2.05, 160)
    for harmonic, color, label in ((1, "#343a40", "X1"), (2, "#8d5bb5", "X2")):
        Bres = resonance_field_T(105.0e9, harmonic)
        Rres = Bt_scan * R0 / Bres
        ax.plot(Bt_scan, Rres, color=color, lw=2.2, label=f"{label} 共振位置")
    ax.axhspan(R0 - a, R0 + a, color="#dcefe5", alpha=0.75, label="等离子体径向范围")
    ax.axvspan(1.70, 2.00, color="#f0e1cc", alpha=0.55, label="用户给出的常见 $B_t$")
    x2_layer = next(item for item in assessment["layers"] if item["harmonic"] == 2)
    ax.scatter([1.85], [x2_layer["resonance_R_m"]], s=52, marker="o", color="#8d5bb5", zorder=5)
    ax.annotate(
        f"X2: R={x2_layer['resonance_R_m']:.3f} m",
        xy=(1.85, x2_layer["resonance_R_m"]), xytext=(1.72, 1.20),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    ax.set(xlabel="$B_t(R_0)$ [T]", ylabel="共振层 R [m]", title="磁场扫描：X2 在 1.7-2.0 T 内均可达")
    ax.set_ylim(0.35, 1.43)
    style_axis(ax)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle("J-TEXT 代表性 pre-CFS 几何诊断（无 GS/EFIT）", fontsize=14, weight="bold")
    output = FIGURE_DIR / "jtext_resonance_accessibility.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def run_relativistic_demo() -> dict[str, object]:
    grid = RelativisticMomentumGrid.uniform(61, 44, 0.8, 0.8)
    solver = RelativisticFokkerPlanck2V(grid, cross_section_area_m2=0.2)
    initial = solver.maxwellian_state(1.0e17, 2.0e3)
    diffusion = QuasilinearDiffusion2V.gaussian_resonance(
        grid,
        peak_D_parallel_m2_s3=1.0e21,
        peak_D_perp_m2_s3=2.0e21,
        resonant_v_parallel_m_s=6.0e7,
        resonant_v_perp_m_s=8.0e7,
        sigma_v_parallel_m_s=2.0e7,
        sigma_v_perp_m_s=2.0e7,
        cross_fraction=0.30,
    )
    state = initial
    time_s = [0.0]
    current_A = [solver.moments(state).current_parallel_A]
    temperature_eV = [solver.moments(state).temperature_eV]
    for step in range(100):
        state, diagnostics = solver.advance(
            state,
            1.0e-9,
            collision_frequency_s=2.0e5,
            collision_temperature_eV=2.0e3,
            rf_diffusion=diffusion,
            momentum_conserving_field_particle=True,
        )
        time_s.append((step + 1) * 1.0e-9)
        moments = solver.moments(state)
        current_A.append(moments.current_parallel_A)
        temperature_eV.append(moments.temperature_eV)
    return {
        "grid": grid,
        "solver": solver,
        "initial": initial,
        "final": state,
        "diffusion": diffusion,
        "time_s": np.asarray(time_s),
        "current_A": np.asarray(current_A),
        "temperature_eV": np.asarray(temperature_eV),
        "diagnostics": diagnostics,
    }


def plot_relativistic_demo(result: dict[str, object]) -> Path:
    grid = result["grid"]
    initial = result["initial"]
    final = result["final"]
    diffusion = result["diffusion"]
    up, ut = grid.mesh_u()
    f0 = initial.f_e
    f1 = final.f_e
    floor = max(float(np.max(f0)), float(np.max(f1))) * 1.0e-10
    delta = (f1 - f0) / max(float(np.max(f0)), 1.0e-300)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7), constrained_layout=True)
    for ax, field, title in (
        (axes[0, 0], f0, "初始 Juttner 分布"),
        (axes[0, 1], f1, "100 ns 后分布"),
    ):
        image = ax.pcolormesh(up, ut, np.maximum(field, floor), shading="auto", cmap="magma", norm=LogNorm(vmin=floor, vmax=max(float(np.max(f0)), float(np.max(f1)))))
        ax.contour(up, ut, field, levels=np.geomspace(max(float(np.max(field)) * 1e-8, floor), float(np.max(field)), 8), colors="black", linewidths=0.6, alpha=0.9)
        ax.set(xlabel=r"$u_\parallel=p_\parallel/(m_ec)$", ylabel=r"$u_\perp=p_\perp/(m_ec)$", title=title)
        fig.colorbar(image, ax=ax, label=r"$f_e$ [SI, log]")

    limit = max(abs(float(np.min(delta))), abs(float(np.max(delta))), 1.0e-12)
    image = axes[1, 0].pcolormesh(up, ut, delta, shading="auto", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
    axes[1, 0].contour(up, ut, diffusion.D_cross_m2_s3, levels=6, colors="#222222", linewidths=0.55)
    axes[1, 0].set(xlabel=r"$u_\parallel$", ylabel=r"$u_\perp$", title=r"分布变化；黑线为非对角 $D_{\parallel\perp}$")
    fig.colorbar(image, ax=axes[1, 0], label=r"$(f-f_0)/\max(f_0)$")

    ax = axes[1, 1]
    ax.plot(result["time_s"] * 1.0e9, result["current_A"], color="#2b6f9f", lw=2.2, label="平行电流")
    ax.set(xlabel="时间 [ns]", ylabel="平行电流 [A]", title="非对称 Dql 产生的分布矩电流")
    style_axis(ax)
    ax2 = ax.twinx()
    ax2.plot(result["time_s"] * 1.0e9, result["temperature_eV"], color="#b55239", lw=1.7, ls="--", label="动理学温度")
    ax2.set_ylabel("动理学温度 [eV]")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], loc="center right", frameon=False)
    fig.suptitle("相对论动量 F-P + 场粒子项 + 非对角 Dql（合成验证，不是 J-TEXT 定量预测）", fontsize=13, weight="bold")
    output = FIGURE_DIR / "relativistic_fp_offdiagonal_dql.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_orbit_average(result: dict[str, object]) -> Path:
    grid = result["grid"]
    initial = result["initial"]
    vp, vt = grid.mesh()
    up, ut = grid.mesh_u()
    mirror_ratios = np.linspace(1.0, 2.5, 100)
    fractions = []
    distribution_weights = initial.f_e * grid.phase_volume_weights_m3_s3
    for ratio in mirror_ratios:
        fractions.append(classify_pitch(vp, vt, B_min_T=1.0, B_max_T=float(ratio), weights=distribution_weights).trapped_fraction)
    selected = classify_pitch(vp, vt, B_min_T=1.0, B_max_T=1.8, weights=distribution_weights)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), constrained_layout=True)
    image = axes[0].pcolormesh(up, ut, selected.trapped_mask.astype(float), shading="auto", cmap="Set2", vmin=0, vmax=1)
    axes[0].contour(up, ut, selected.trapped_mask.astype(float), levels=[0.5], colors="#222222", linewidths=1.2)
    axes[0].set(xlabel=r"$u_\parallel$", ylabel=r"$u_\perp$", title="M=1.8：通过/俘获分类")
    cbar = fig.colorbar(image, ax=axes[0], ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["通过", "俘获"])

    image = axes[1].pcolormesh(up, ut, selected.bounce_weight, shading="auto", cmap="viridis")
    axes[1].contour(up, ut, selected.trapped_mask.astype(float), levels=[0.5], colors="black", linewidths=1.1)
    axes[1].set(xlabel=r"$u_\parallel$", ylabel=r"$u_\perp$", title="驻留时间 bounce 权重")
    fig.colorbar(image, ax=axes[1], label="无量纲权重")

    axes[2].plot(mirror_ratios, 100.0 * np.asarray(fractions), color="#1f8a70", lw=2.3)
    axes[2].scatter([1.8], [100.0 * selected.trapped_fraction], color="#b55239", s=45, zorder=4)
    axes[2].annotate(f"M=1.8: {100*selected.trapped_fraction:.1f}%", xy=(1.8, 100*selected.trapped_fraction), xytext=(1.25, 72), arrowprops={"arrowstyle": "->", "color": "#555555"})
    axes[2].set(xlabel="镜像比 M=Bmax/Bmin", ylabel="分布加权俘获比例 [%]", title="俘获比例对镜像比的响应")
    axes[2].set_ylim(0, 100)
    style_axis(axes[2])
    fig.suptitle("pre-CFS 俘获/通过粒子与 bounce-average 镜像代理", fontsize=13, weight="bold")
    output = FIGURE_DIR / "trapped_passing_bounce_average.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_exl50u_overview() -> Path:
    """Reuse the formal EXL-50U case comparison as the advanced-report overview."""
    image = plt.imread(EXL_DIR / "exl50u_case_comparison.png")
    fig, axis = plt.subplots(figsize=(11.3, 5.1), constrained_layout=True)
    axis.imshow(image)
    axis.axis("off")
    fig.suptitle("EXL-50U 0--100 ms：EC source/CS comparison", fontsize=13, weight="bold")
    output = FIGURE_DIR / "exl50u_case_comparison.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def report_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCN", parent=styles["Title"], fontName="Deng", fontSize=22, leading=29, textColor=colors.HexColor("#17324d"), alignment=TA_LEFT, spaceAfter=7 * mm),
        "subtitle": ParagraphStyle("SubtitleCN", parent=styles["Normal"], fontName="Deng", fontSize=11, leading=17, textColor=colors.HexColor("#4d5963"), spaceAfter=7 * mm),
        "h1": ParagraphStyle("H1CN", parent=styles["Heading1"], fontName="Deng", fontSize=16, leading=22, textColor=colors.HexColor("#17324d"), spaceBefore=2 * mm, spaceAfter=4 * mm),
        "body": ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName="Deng", fontSize=9.5, leading=15, textColor=colors.HexColor("#22282d"), spaceAfter=3 * mm),
        "caption": ParagraphStyle("CaptionCN", parent=styles["BodyText"], fontName="Deng", fontSize=8.5, leading=13, textColor=colors.HexColor("#59636b"), alignment=TA_CENTER, spaceBefore=2 * mm, spaceAfter=3 * mm),
        "small": ParagraphStyle("SmallCN", parent=styles["BodyText"], fontName="Deng", fontSize=8.2, leading=12.5, textColor=colors.HexColor("#59636b")),
    }


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Deng", 8)
    canvas.setFillColor(colors.HexColor("#68727a"))
    canvas.drawString(18 * mm, 11 * mm, "linEC 高级动理学算例")
    canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_pdf(figures: dict[str, Path], demo: dict[str, object]) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="linEC EXL-50U startup and advanced kinetic assessment",
        author="linEC",
    )
    styles = report_styles()
    story = [
        Paragraph("linEC EXL-50U 启动与高级动理学诊断", styles["title"]),
        Paragraph(
        "本报告汇总 EXL-50U 0--100 ms 启动案例以及相对论动量网格、线性化动量守恒场粒子项、非对角 Dql、俘获/通过粒子和 pre-CFS bounce-average 镜像代理。所有结果按证据层级标记，避免把 surrogate 敏感性结果误写成无感启动实验验证。",
            styles["subtitle"],
        ),
    ]
    summary_data = [
        [Paragraph("结果类别", styles["small"]), Paragraph("用途", styles["small"]), Paragraph("定量边界", styles["small"])],
        [Paragraph("EXL-50U 0--100 ms", styles["small"]), Paragraph("28/50 GHz、CS/no-CS、F-P 反馈与阈值敏感性", styles["small"]), Paragraph("pre-CFS surrogate，待炮级标定", styles["small"])],
        [Paragraph("相对论 F-P", styles["small"]), Paragraph("验证 u 网格、场粒子项和非对角 Dql", styles["small"]), Paragraph("合成 Dql，不是某炮预测", styles["small"])],
        [Paragraph("轨道平均", styles["small"]), Paragraph("验证 trapped/passing 与镜像响应", styles["small"]), Paragraph("pre-CFS surrogate，不是真实闭合轨道", styles["small"])],
        [Paragraph("J-TEXT 105 GHz", styles["small"]), Paragraph("常规托卡马克 X1/X2 几何反例", styles["small"]), Paragraph("仅几何诊断，无启动定量结论", styles["small"])],
    ]
    table = Table(summary_data, colWidths=[40 * mm, 70 * mm, 55 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Deng"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6ee")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bac3ca")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([table, Spacer(1, 7 * mm), Paragraph("当前证据等级", styles["h1"]), Paragraph(
        "EXL-50U 六组 0--100 ms 曲线和功率/压强阈值目前属于机理与敏感性结果。由于尚缺炮级 PF/TF/CS/TPC/VFC 波形、标定波束与功率归一化 Dql、壁/轨道损失以及成功/失败炮和 Ip(t) 验证，质量门不允许把图中的启动状态或电流值称为设备级定量预测。",
        styles["body"],
    ), PageBreak()])

    story.extend([Paragraph("1. EXL-50U 0--100 ms 启动案例", styles["h1"]), Image(str(figures["exl"]), width=174 * mm, height=76 * mm), Paragraph(
        "研究主线是 EXL-50U 启动段：28 GHz O、50 GHz X 与联合注入分别比较 CS/no-CS，并把电子分布、F-P 电流矩、阶段机制和压力/功率阈值放在同一工作流。0--100 ms 不使用 GS/EFIT 磁面。",
        styles["caption"],
    ), Paragraph("设备级质量门", styles["h1"]), Paragraph(
        "跨装置要给出“能否启动”和 Ip 大小，必须输入实际线圈波形、气压与杂质、标定 EC 波束/Dql、壁与轨道损失，并通过独立成功/失败炮分类和 Ip 时序误差检验。",
        styles["body"],
    ), PageBreak()])

    final_moments = demo["solver"].moments(demo["final"])
    story.extend([Paragraph("2. 相对论 F-P 与非对角 Dql", styles["h1"]), Image(str(figures["fp"]), width=174 * mm, height=128 * mm), Paragraph(
        f"该合成算例使用 u=p/(m_e c) 网格、场粒子项和 cross_fraction=0.30 的正半定扩散张量。100 ns 后平行电流矩为 {final_moments.current_parallel_A:.1f} A，动理学温度为 {final_moments.temperature_eV:.1f} eV。数值只用于验证响应方向、守恒和耦合接口，不能当作 J-TEXT 或 EXL-50U 的电流预测。",
        styles["caption"],
    ), PageBreak()])

    story.extend([Paragraph("3. 俘获/通过粒子和 bounce-average 代理", styles["h1"]), Image(str(figures["orbit"]), width=174 * mm, height=61 * mm), Paragraph(
        "分类采用 lambda Bmax>=1；bounce 权重代表沿轨道的驻留时间。镜像比升高时，分布加权俘获比例上升。这能表达 TPC 类镜像位形相对普通 VFC/circular 开场线的约束差异，但没有真实 PF 线圈、壁命中轨道和闭合磁面时，不能称为 CQL3D 式定量 bounce average。",
        styles["caption"],
    ), Paragraph("启动阶段应如何使用", styles["h1"]), Paragraph(
        "0-100 ms 且 CFS 尚未形成时，输入实际 PF/TF/CS/TPC/VFC 波形和壁几何，由场线追踪得到 Bmin/Bmax、连接长度和壁命中损失，再把这些量送入本模块。CFS 形成后，才切换到真实磁面/导引中心轨道平均。",
        styles["body"],
    ), PageBreak()])

    story.extend([Paragraph("4. 常规托卡马克几何诊断附录：J-TEXT", styles["h1"]), Image(str(figures["jtext"]), width=174 * mm, height=72 * mm), Paragraph(
        "在 Bt=1.7--2.0 T 的代表性 J-TEXT 几何中，105 GHz X2 层仍可位于等离子体径向范围。因此，“缺少二次谐波层”不能单独解释无感启动失败；还需检验冷启动吸收、平行动量、开场线约束和气压碰撞。该页只回答共振层几何可达性。",
        styles["caption"],
    ), Paragraph("跨装置解释边界", styles["h1"]), Paragraph(
        "低纵横比、高比压和多共振层必须分别由 R0/a 扫描、压力演化和波/轨道计算产生，不能写成 ST 标签乘子；每个真实装置仍需单独标定和验证。",
        styles["body"],
    ), Spacer(1, 4 * mm), Paragraph(
        "参考：Xia et al., EPJ Web of Conferences 313, 02003 (2024), DOI 10.1051/epjconf/202431302003；Harvey and McCoy, CQL3D (1993)；Lin-Liu et al., Physics of Plasmas 10 (2003), DOI 10.1063/1.1610472。",
        styles["small"],
    )])
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    demo = run_relativistic_demo()
    figures = {
        "jtext": plot_jtext_resonance(),
        "fp": plot_relativistic_demo(demo),
        "orbit": plot_orbit_average(demo),
        "exl": plot_exl50u_overview(),
    }
    build_pdf(figures, demo)
    print(PDF_PATH)
    for path in figures.values():
        print(path)


if __name__ == "__main__":
    main()
