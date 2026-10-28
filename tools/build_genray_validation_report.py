"""Build the audited GENRAY integration/validation PDF appendix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    if regular.exists():
        pdfmetrics.registerFont(TTFont("ReportCJK", regular))
        pdfmetrics.registerFont(TTFont("ReportCJKBold", bold if bold.exists() else regular))
        return "ReportCJK", "ReportCJKBold"
    return "Helvetica", "Helvetica-Bold"


def _fit_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as bitmap:
        width, height = bitmap.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _dql_figure(npz_path: Path, output: Path) -> None:
    with np.load(npz_path) as archive:
        parallel = archive["v_parallel_m_s"] / 1.0e6
        perpendicular = archive["v_perp_m_s"] / 1.0e6
        dpar = archive["D_parallel_m2_s3"]
        dperp = archive["D_perp_m2_s3"]
    floor = max(float(max(np.max(dpar), np.max(dperp))) * 1.0e-10, np.finfo(float).tiny)
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    for axis, data, title in zip(
        axes,
        (dpar, dperp),
        (r"$D_{\parallel\parallel}$", r"$D_{\perp\perp}$"),
    ):
        image = axis.contourf(
            parallel,
            perpendicular,
            np.log10(np.maximum(data, floor)).T,
            levels=25,
            cmap="magma",
        )
        axis.contour(
            parallel,
            perpendicular,
            np.log10(np.maximum(data, floor)).T,
            levels=8,
            colors="white",
            linewidths=0.45,
            alpha=0.65,
        )
        axis.set(xlabel=r"$v_\parallel$ [$10^6$ m/s]", ylabel=r"$v_\perp$ [$10^6$ m/s]", title=title)
        figure.colorbar(image, ax=axis, label=r"$\log_{10} D$ [m$^2$ s$^{-3}$]")
    figure.suptitle("linEC internal GENRAY-like, power-normalized diagonal Dql")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_report(run_directory: Path, output_pdf: Path) -> None:
    validation = json.loads((run_directory / "validation_comparison.json").read_text(encoding="utf-8"))
    native = json.loads((run_directory / "genray_native_summary.json").read_text(encoding="utf-8"))
    internal = json.loads((run_directory / "internal_dql_summary.json").read_text(encoding="utf-8"))
    fp = json.loads((run_directory / "fp_summary.json").read_text(encoding="utf-8"))
    scratch = Path("tmp/pdfs/genray_validation")
    dql_plot = scratch / "internal_dql.png"
    _dql_figure(run_directory / "internal_dql.npz", dql_plot)

    font, bold = _register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCJK", parent=styles["Title"], fontName=bold, fontSize=20, leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#133c55"))
    heading = ParagraphStyle("HeadingCJK", parent=styles["Heading2"], fontName=bold, fontSize=14, leading=20, textColor=colors.HexColor("#15616d"), spaceAfter=6)
    body = ParagraphStyle("BodyCJK", parent=styles["BodyText"], fontName=font, fontSize=9.2, leading=14, textColor=colors.HexColor("#263238"))
    small = ParagraphStyle("SmallCJK", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#455a64"))
    warning = ParagraphStyle("WarningCJK", parent=body, fontName=bold, textColor=colors.HexColor("#9b2226"), backColor=colors.HexColor("#fff1f0"), borderPadding=7)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_pdf), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="linEC GENRAY integration validation report",
        author="linEC / ecstartup",
    )
    story = [
        Paragraph("linEC GENRAY 接口与内部降阶模型验证报告", title),
        Spacer(1, 5 * mm),
        Paragraph(
            "本报告区分三条链：原生 GENRAY NetCDF 读取、外部功率归一化 Dql 交换、"
            "以及 linEC 内部冷 Stix 射线 + 相对论谐波核 + 0D-2V F-P。官方算例来自 "
            f"compxco/genray commit {validation.get('upstream_commit') or '未记录'}。",
            body,
        ),
        Spacer(1, 4 * mm),
        Paragraph("结论", heading),
    ]
    table_data = [
        ["检查项", "结果", "判定"],
        ["原生吸收功率", f"{native['absorbed_power_W']/1e3:.3f} kW", "通过"],
        ["原生平行电流", f"{native['parallel_current_A']/1e3:.3f} kA", "基准"],
        ["linEC 伴随量级", f"{validation['linec_adjoint_scale_parallel_current_A']/1e3:.3f} kA", "2 倍内"],
        ["linEC 0D-2V F-P", f"{validation['linec_fp_parallel_current_A']/1e3:.2f} kA", "失败"],
        ["功率守恒误差", f"{native['power_balance_relative']:.2e}", "通过"],
    ]
    table = Table(table_data, colWidths=[62 * mm, 55 * mm, 43 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), bold),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#15616d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fafb")),
        ("TEXTCOLOR", (2, 4), (2, 4), colors.HexColor("#9b2226")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [table, Spacer(1, 4 * mm), Paragraph(
        "0D-2V F-P 电流比 GENRAY 基准大约 "
        f"{validation['linec_fp_to_reference_abs_ratio']:.1f} 倍。该差异被保留为失败质量门；"
        "没有通过修改碰撞频率、面积或归一化把结果强行拟合到基准。",
        warning,
    ), Spacer(1, 4 * mm), _fit_image(run_directory / "current_validation.png", 175 * mm, 85 * mm),
        PageBreak(), Paragraph("1. 原生 GENRAY NetCDF 回归", heading),
        Paragraph(
            "适配器按 nrayelt 截断每条射线，并完成 cm、gauss、particles/cm3、erg/s、"
            "A/cm2 到 SI 的转换。标准 genray.nc 提供射线、沉积和总电流，但不提供本项目 "
            "F-P 网格所需的二维速度空间 Dql。",
            body,
        ), Spacer(1, 3 * mm), _fit_image(run_directory / "genray_native_validation.png", 176 * mm, 178 * mm),
        Spacer(1, 2 * mm), Paragraph(
            f"170 GHz O1；射线 {native['rays'][0]['sample_count']} 点，路径 "
            f"{native['rays'][0]['path_length_m']:.3f} m；注入 {native['injected_power_W']/1e3:.3f} kW，"
            f"吸收率 {100*native['absorption_fraction']:.5f}%。",
            small,
        ), PageBreak(), Paragraph("2. 内部 GENRAY-like Dql", heading),
        Paragraph(
            "内部后端在无磁面 MagneticFieldProvider 上推进冷等离子体 Hamilton 射线，以 "
            "omega-k_parallel v_parallel-s Omega_e/gamma 的有限宽共振核映射到速度空间，"
            "再用初始 Maxwellian 的能量矩把 Dql 归一化到参考吸收功率。",
            body,
        ), Spacer(1, 3 * mm), _fit_image(dql_plot, 176 * mm, 90 * mm), Spacer(1, 3 * mm),
    ]
    diag = internal["diagnostics"][0]
    diag_data = [
        ["内部诊断", "值"],
        ["射线状态 / 点数", f"{diag['ray_status']} / {diag['ray_sample_count']}"],
        ["内部射线路径", f"{diag['ray_path_length_m']:.3f} m"],
        ["参考吸收功率", f"{diag['target_absorbed_power_W']/1e3:.3f} kW"],
        ["降阶预测吸收率", f"{100*diag['predicted_absorption_fraction']:.3f}%"],
        ["速度网格共振覆盖", f"{100*diag['resonance_velocity_coverage_fraction']:.2f}%"],
        ["最低 WKB 尺度比", f"{diag['minimum_wkb_scale_ratio']:.1f}"],
        ["zeta / 伴随电流量级", f"{diag['adjoint_normalized_efficiency']:.2f} / {diag['adjoint_estimated_parallel_current_A']/1e3:.3f} kA"],
    ]
    diag_table = Table(diag_data, colWidths=[80 * mm, 80 * mm])
    diag_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), bold),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5f3f2")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b0bec5")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [diag_table, PageBreak(), Paragraph("3. 电子分布与未通过的电流闭合", heading),
        _fit_image(run_directory / "fp_distribution_evolution.png", 176 * mm, 175 * mm),
        Spacer(1, 2 * mm), Paragraph(
            f"F-P 终态时间 {fp['time_s']*1e3:.1f} ms，终态平行电流 "
            f"{fp['final_moments']['current_parallel_A']/1e3:.2f} kA，最大粒子守恒相对误差 "
            f"{fp['max_abs_particle_balance_relative']:.2e}。数值守恒通过，但电流物理量级未通过。",
            small,
        ), Spacer(1, 3 * mm), Paragraph("当前缺失项", heading), Paragraph(
            "要把内部 F-P 电流升级为可与 GENRAY/CQL3D 定量对照的结果，至少需要：相对论动量网格；"
            "测试粒子与场粒子的线性化 Coulomb 算子及动量守恒；非对角准线性张量；俘获/通过粒子和 "
            "真实导引中心 bounce average（现有实现只是 pre-CFS 镜像代理）；空间沉积到环向电流的磁场投影；以及伴随电流效率。CFS 前仍不得调用 "
            "GS/EFIT 磁面，CFS 形成后才允许切换到磁面坐标后端。",
            warning,
        ), Spacer(1, 4 * mm), Paragraph("参考", heading), Paragraph(
            "Smirnov, Harvey & Prater, DOI 10.1142/9789812814647_0042；Lin-Liu et al., "
            "DOI 10.1063/1.1610472；Marushchenko et al., DOI 10.1088/0029-5515/48/5/054002。",
            small,
        )]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7.5)
        canvas.setFillColor(colors.HexColor("#607d8b"))
        canvas.drawString(16 * mm, 8 * mm, "linEC / ecstartup - GENRAY validation appendix")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    build_report(args.run_directory, args.output_pdf)


if __name__ == "__main__":
    main()
