"""EXL-50U 0--100 ms source/CS comparison and unified report workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import SimulationConfig
from .current_profile import reconstruct_current_profiles, reconstruct_mechanism_currents
from .device_prediction_quality import assess_device_prediction_quality
from .io import write_run
from .reporting import _draw_profile_page, _pyplot, _report_threshold_scan_pages
from .solver import RunResult, integrate
from .stage_mechanisms import MECHANISM_LABELS, summarize_stage_mechanisms
from .threshold_scan import run_threshold_scan


@dataclass(frozen=True)
class EXLCase:
    name: str
    source_label: str
    cs_mode: str
    config: SimulationConfig
    result: RunResult


def _source_case_config(
    base: SimulationConfig,
    enabled_sources: Iterable[str],
    cs_mode: str,
) -> SimulationConfig:
    enabled = set(enabled_sources)
    raw = base.model_dump(mode="python")
    available = {source["name"] for source in raw["rf"]["sources"]}
    unknown = sorted(enabled - available)
    if unknown:
        raise ValueError(f"unknown RF sources: {', '.join(unknown)}")
    for source in raw["rf"]["sources"]:
        source["enabled"] = source["name"] in enabled
    raw["circuit"]["cs_mode"] = cs_mode
    raw["threshold_scan"]["enabled"] = False
    raw["outputs"]["write_current_profile"] = False
    raw["outputs"]["write_pdf_report"] = False
    return SimulationConfig.model_validate(raw)


def _threshold_config(base: SimulationConfig, source_name: str) -> SimulationConfig:
    raw = base.model_dump(mode="python")
    raw["threshold_scan"]["enabled"] = True
    raw["threshold_scan"]["source_name"] = source_name
    raw["outputs"]["write_current_profile"] = False
    raw["outputs"]["write_pdf_report"] = False
    return SimulationConfig.model_validate(raw)


def _case_summary(case: EXLCase) -> dict:
    rows = case.result.diagnostics
    peak = max(rows, key=lambda row: abs(float(row.get("Ip_A", 0.0))))
    event_types = {event.type for event in case.result.events}
    return {
        "name": case.name,
        "source": case.source_label,
        "cs_mode": case.cs_mode,
        "solver_success": case.result.success,
        "startup_criterion_reached": "cfs" in event_types,
        "startup_criterion": "configured CFS event",
        "startup_classification_is_validated": False,
        "final_stage": str(rows[-1].get("stage", "unknown")),
        "peak_Ip_A": float(peak.get("Ip_A", 0.0)),
        "peak_ne_m3": max(float(row.get("ne_m3", 0.0)) for row in rows),
        "peak_Te_eV": max(float(row.get("Te_eV", 0.0)) for row in rows),
        "peak_I_kinetic_A": max(abs(float(row.get("I_kinetic_A", 0.0))) for row in rows),
        "kinetic_coupling_mode": case.config.kinetic.coupling_mode,
        "events": [{"type": event.type, "time_s": event.time_s} for event in case.result.events],
    }


def _write_stage_mechanism_pages(pdf, cases: list[EXLCase], summaries: list[dict]) -> None:
    plt = _pyplot()
    for cs_mode, title in (
        ("without_cs", "weak/no-CS experimental category"),
        ("with_cs", "prescribed 5 V loop-voltage comparison"),
    ):
        selected = [item for item in summaries if item["cs_mode"] == cs_mode]
        fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        axis = fig.add_subplot(111)
        axis.axis("off")
        rows = []
        for item in selected:
            for window in item["windows"]:
                if window["time_range_s"] is None:
                    interval = "not reached"
                else:
                    interval = "{:.1f}--{:.1f}".format(
                        1.0e3 * window["time_range_s"][0],
                        1.0e3 * window["time_range_s"][1],
                    )
                dominant = window["dominant_label"]
                fraction = window.get("mechanism_fraction", {}).get(
                    window.get("dominant_mechanism", ""), 0.0
                )
                rows.append(
                    [
                        item["case"],
                        window["label"],
                        interval,
                        "yes" if window["terminal_criterion_reached"] else "no",
                        dominant,
                        f"{100.0 * fraction:.1f}%" if dominant != "not reached" else "--",
                    ]
                )
        table = axis.table(
            cellText=rows,
            colLabels=["case", "diagnostic window", "time [ms]", "end criterion", "dominant CD", "share"],
            loc="center",
            cellLoc="left",
            colWidths=[0.16, 0.25, 0.12, 0.12, 0.25, 0.08],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1.0, 1.45)
        axis.set_title(f"Stage-resolved causal current contributions | {title}", fontsize=14, pad=16)
        axis.text(
            0.01,
            0.02,
            "Shares use mean absolute causal L-R contributions. Numerical closure residual is excluded. "
            "'Not reached' means the configured density/temperature criterion was not achieved.",
            transform=axis.transAxes,
            fontsize=9,
        )
        pdf.savefig(fig)
        plt.close(fig)


def _write_mechanism_share_page(pdf, cases: list[EXLCase]) -> None:
    plt = _pyplot()
    selected = [case for case in cases if case.source_label == "50 GHz X"]
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), constrained_layout=True)
    colors = ["#2563eb", "#dc2626", "#ca8a04", "#059669"]
    for axis, case in zip(axes, selected):
        currents = reconstruct_mechanism_currents(case.result, case.config)
        values = np.vstack(
            [
                np.abs(currents.I_cs_A),
                np.abs(currents.I_rf_A),
                np.abs(currents.I_tail_A),
                np.abs(currents.I_vertical_A),
            ]
        )
        total = np.sum(values, axis=0)
        shares = np.divide(values, total, out=np.zeros_like(values), where=total > 0.0)
        axis.stackplot(
            case.result.times_s * 1.0e3,
            shares,
            labels=[MECHANISM_LABELS[key] for key in ("cs_inductive", "fp_ec_moment", "hot_tail", "vertical_charge")],
            colors=colors,
            alpha=0.85,
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("absolute-current share")
        axis.set_xlabel("time [ms]")
        axis.set_title(case.name)
        axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper center")
    fig.suptitle("Time-resolved dominant current-drive mechanism | 50 GHz X-mode", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


def _add_semicircle_velocity_grid(axis, maximum_speed: float) -> None:
    theta = np.linspace(0.0, np.pi, 241)
    for speed in np.linspace(maximum_speed / 4.0, maximum_speed, 4):
        axis.plot(speed * np.cos(theta), speed * np.sin(theta), color="black", linewidth=0.45, linestyle=(0, (2, 3)), alpha=0.3)
    for angle_deg in (30.0, 60.0, 90.0, 120.0, 150.0):
        angle = np.deg2rad(angle_deg)
        axis.plot([0.0, maximum_speed * np.cos(angle)], [0.0, maximum_speed * np.sin(angle)], color="black", linewidth=0.4, linestyle=(0, (2, 4)), alpha=0.22)


def _write_kinetic_distribution_pages(pdf, cases: list[EXLCase]) -> None:
    plt = _pyplot()
    for case in [item for item in cases if item.source_label == "50 GHz X"]:
        history = case.result.kinetic_history
        if history is None or len(history.snapshot_times_s) == 0:
            continue
        indices = np.unique(np.asarray([0, len(history.snapshot_times_s) // 2, len(history.snapshot_times_s) - 1], dtype=int))
        snapshots = history.snapshot_f_e[indices]
        floor = max(float(np.max(snapshots)) * 1.0e-13, np.finfo(float).tiny)
        logs = np.log10(np.maximum(snapshots, floor))
        levels = np.linspace(float(np.min(logs)), float(np.max(logs)), 25)
        vpar = history.v_parallel_m_s / 1.0e6
        vperp = history.v_perp_m_s / 1.0e6
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
        image = None
        for axis, source_index, values in zip(axes.flat[:3], indices, logs):
            image = axis.contourf(vpar, vperp, values.T, levels=levels, cmap="viridis", extend="both")
            internal_levels = levels[(levels > float(np.min(values))) & (levels < float(np.max(values)))][::4]
            if internal_levels.size:
                axis.contour(vpar, vperp, values.T, levels=internal_levels, colors="black", linewidths=0.65, alpha=0.9)
            time_s = float(history.snapshot_times_s[source_index])
            current = float(np.interp(time_s, history.times_s, history.incremental_current_A))
            axis.set_title(f"t={1e3*time_s:.1f} ms | F-P EC current={current/1e3:.3f} kA")
            axis.set_xlabel("$v_\\parallel$ [$10^6$ m/s]")
            axis.set_ylabel("$v_\\perp$ [$10^6$ m/s]")
            maximum = min(abs(float(vpar[0])), float(vpar[-1]), float(vperp[-1]))
            axis.set_xlim(float(vpar[0]), float(vpar[-1]))
            axis.set_ylim(0.0, float(vperp[-1]))
            axis.set_aspect("equal", adjustable="box")
            _add_semicircle_velocity_grid(axis, maximum)
        if image is not None:
            fig.colorbar(image, ax=list(axes.flat[:3]), label="$\\log_{10} f_e$ [SI]")
        axis = axes[1, 1]
        axis.plot(history.times_s * 1.0e3, history.powered_current_A / 1.0e3, label="powered F-P moment")
        axis.plot(history.times_s * 1.0e3, history.zero_dql_current_A / 1.0e3, label="zero-Dql baseline")
        axis.plot(history.times_s * 1.0e3, history.incremental_current_A / 1.0e3, linewidth=2.0, label="EC increment")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel("time [ms]")
        axis.set_ylabel("parallel current moment [kA]")
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False, fontsize=8)
        residual = history.fixed_point_residual_A[-1] if history.fixed_point_residual_A else 0.0
        axis.set_title(f"fixed-point iterations={history.fixed_point_iterations}, last residual={residual:.2f} A")
        fig.suptitle(
            f"Time-dependent electron distribution | {case.name}\n{history.diffusion_source}",
            fontsize=13,
        )
        pdf.savefig(fig)
        plt.close(fig)


def _write_experimental_prior_page(
    pdf, config: SimulationConfig, threshold_payloads: list[dict]
) -> None:
    plt = _pyplot()
    prior = config.experimental_startup_prior
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.10, top=0.88, wspace=0.28)
    axis = axes[0]
    robust_kw = float(prior.robust_startup_power_upper_bound_W) / 1.0e3
    reference_kw = float(prior.robust_reference_power_W) / 1.0e3
    pressure_mpa = float(prior.optimum_pressure_Pa) * 1.0e3
    ceiling_mpa = float(prior.low_power_pressure_ceiling_Pa) * 1.0e3
    xmax = max(320.0, reference_kw * 1.05)
    axis.axvspan(robust_kw, xmax, color="#bbf7d0", alpha=0.45, label="experimentally robust startup region")
    axis.axvline(robust_kw, color="#15803d", linestyle="--", label=f"robust threshold <= {robust_kw:g} kW")
    axis.fill_between([0.0, float(prior.low_power_reference_W) / 1.0e3], ceiling_mpa, 12.0, color="#fecaca", alpha=0.45, label="reported higher termination risk at 70 kW")
    axis.scatter([float(prior.low_power_reference_W) / 1.0e3], [ceiling_mpa], marker="x", s=80, color="#b91c1c", label="70 kW pressure ceiling anchor")
    optimum_powers = np.asarray(prior.optimum_pressure_reference_powers_W, dtype=float) / 1.0e3
    axis.scatter(optimum_powers, np.full_like(optimum_powers, pressure_mpa), s=60, color="#1d4ed8", label="best-Ip pressure anchors")
    axis.axvline(reference_kw, color="#166534", linestyle=":", label=f"{reference_kw:g} kW robust reference")
    model_payload = next(
        (item for item in threshold_payloads if item.get("source_name") == "EC50_X"),
        None,
    )
    if model_payload is not None:
        thresholds = [
            row
            for row in model_payload["thresholds"]
            if row["cs_mode"] == "without_cs"
            and row["minimum_successful_ec_power_W"] is not None
        ]
        for zeff in sorted({float(row["effective_charge"]) for row in thresholds}):
            rows = sorted(
                (row for row in thresholds if float(row["effective_charge"]) == zeff),
                key=lambda row: row["pressure_Pa"],
            )
            axis.plot(
                [float(row["minimum_successful_ec_power_W"]) / 1.0e3 for row in rows],
                [float(row["pressure_Pa"]) * 1.0e3 for row in rows],
                marker="s",
                markersize=4,
                linewidth=1.2,
                linestyle="--",
                label=f"model 1 kA upper bound, Zeff={zeff:g}",
            )
    axis.set_xlim(0.0, xmax)
    axis.set_ylim(0.0, 12.0)
    axis.set_xlabel("injected EC power [kW]")
    axis.set_ylabel("fill pressure [mPa]")
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False, fontsize=7.5, loc="upper right")
    axis.set_title("User-supplied EXL-50U experimental anchors")
    axes[1].axis("off")
    text = (
        f"Weak CS: {float(prior.weak_cs_current_A):g} A coil current, treated as no-CS for this report.\n"
        "No loop voltage is inferred without CS mutual inductance and dI_CS/dt.\n\n"
        f"Priority: {' > '.join(prior.importance_order)}. This is an ordering, not an arbitrary numeric weight.\n\n"
        f"At {float(prior.low_power_reference_W)/1e3:g} kW, {ceiling_mpa:g} mPa is a reported pressure-limit anchor.\n"
        f"At {', '.join(f'{value/1e3:g}' for value in prior.optimum_pressure_reference_powers_W)} kW, "
        f"about {pressure_mpa:g} mPa gives the best Ip; this is an optimum, not necessarily a hard startup boundary.\n\n"
        f"At 125 and {reference_kw:g} kW, startup is reported as robust over ordinary pressure/impurity conditions.\n\n"
        "An H-alpha peak often accompanies termination, but is neither assumed necessary nor sufficient. "
        "Some high-pressure shots terminate without a peak. The present code has no shot-resolved H-alpha input.\n\n"
        "These anchors are a calibration/comparison layer supplied by the user. They do not overwrite the ODE, "
        "and have not yet been verified against raw pressure, H-alpha, impurity and EC timing traces.\n\n"
        "Dashed model curves are the successful upper edge for a 1 kA EC-attributable current increment. "
        "Preionization is not required in this scan, so these curves are not complete startup-success thresholds."
    )
    axes[1].text(0.02, 0.96, text, va="top", fontsize=9.5, linespacing=1.35, wrap=True)
    fig.suptitle("Experimental prior versus model threshold interpretation", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


def _write_quality_gate_page(pdf, config: SimulationConfig) -> dict:
    """Add an explicit evidence page and return the machine-readable gate."""

    gate = assess_device_prediction_quality(config)
    plt = _pyplot()
    fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    axis = fig.add_subplot(111)
    axis.axis("off")
    startup_status = "AUTHORIZED" if gate.startup_decision_authorized else "NOT AUTHORIZED"
    current_status = "AUTHORIZED" if gate.current_prediction_authorized else "NOT AUTHORIZED"
    axis.text(0.5, 0.94, "Device-level quantitative prediction gate", ha="center", fontsize=14, weight="bold")
    axis.text(
        0.5,
        0.87,
        f"{gate.device_name} | level={gate.prediction_level} | startup={startup_status} | Ip(t)={current_status}",
        ha="center",
        fontsize=9.5,
        color="#166534" if gate.current_prediction_authorized else "#b91c1c",
    )
    rows = [
        ["Physics inputs", "yes" if gate.physics_inputs_complete else "no"],
        ["Validation data", "yes" if gate.validation_dataset_complete else "no"],
        ["Startup decision", "yes" if gate.startup_decision_authorized else "no"],
        ["Ip(t) prediction", "yes" if gate.current_prediction_authorized else "no"],
    ]
    table = axis.table(cellText=rows, colLabels=["Gate", "Result"], loc="upper center", bbox=[0.22, 0.58, 0.56, 0.19], cellLoc="left", colWidths=[0.38, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#bac3ca")
    axis.text(0.06, 0.50, "Missing device inputs:", fontsize=9.5, weight="bold")
    axis.text(0.07, 0.465, "\n".join(f"- {item}" for item in gate.missing_inputs) or "- none", va="top", fontsize=7.7)
    axis.text(0.53, 0.50, "Failed validation metrics:", fontsize=9.5, weight="bold")
    axis.text(0.54, 0.465, "\n".join(f"- {item}" for item in gate.failed_metrics) or "- none", va="top", fontsize=7.7)
    axis.text(0.06, 0.15, gate.interpretation, va="top", fontsize=9.0, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)
    return gate.to_dict()


def _write_suite_pdf(
    cases: list[EXLCase],
    threshold_payloads: list[dict],
    stage_summaries: list[dict],
    base_config: SimulationConfig,
    path: Path,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        axis = fig.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.92, "EXL-50U EC startup: 0--100 ms model report", ha="center", fontsize=20)
        body = (
            "Cases: 28 GHz O-mode, 50 GHz X-mode, and simultaneous injection; each with and without CS.\n\n"
            "Magnetic treatment: analytic pre-CFS field provider only. No GS/EFIT flux surfaces are used in 0--100 ms.\n"
            "The 500 ms and 102 ms G-EQDSK files are equilibrium references, not startup equilibria.\n\n"
            "Current closure: reduced 0D-2V Fokker--Planck distribution-moment response plus an independent\n"
            "open-field hot-tail preferential-loss surrogate. The configured Gaussian D_ql is synthetic and is not\n"
            "GENRAY/full-wave power-normalized; current and threshold values are sensitivity results, not predictions.\n\n"
            "Thresholds are numerical failure/success brackets relative to a matching zero-EC baseline.\n"
            "With CS, the EC increment includes conductivity-mediated inductive synergy and is not direct ECCD."
        )
        axis.text(0.06, 0.77, body, va="top", fontsize=12, linespacing=1.6)
        axis.text(0.06, 0.16, "Device inputs: B_t=0.83 T; EC28 launch=(1.606, 0.795) m; EC50 launch=(1.725, 0.985) m.", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        _write_quality_gate_page(pdf, base_config)

        _write_stage_mechanism_pages(pdf, cases, stage_summaries)
        _write_mechanism_share_page(pdf, cases)
        _write_kinetic_distribution_pages(pdf, cases)

        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
        fields = (
            ("Ip_A", 1.0e3, "$I_p$ [kA]"),
            ("ne_m3", 1.0, "$n_e$ [m$^{-3}$]"),
            ("Te_eV", 1.0, "$T_e$ [eV]"),
            ("rf_power_absorbed_W", 1.0e3, "absorbed EC [kW]"),
        )
        for axis, (key, scale, ylabel) in zip(axes.flat, fields):
            for case in cases:
                values = np.asarray([float(row.get(key, 0.0)) for row in case.result.diagnostics]) / scale
                axis.plot(case.result.times_s * 1.0e3, values, label=case.name, linewidth=1.25)
            axis.set_xlabel("time [ms]")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.35)
            if key == "ne_m3":
                axis.set_yscale("log")
        axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
        fig.suptitle("Source and CS comparison", fontsize=15)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27), constrained_layout=True)
        for axis, case in zip(axes.flat, cases):
            currents = reconstruct_mechanism_currents(case.result, case.config)
            time_ms = case.result.times_s * 1.0e3
            axis.plot(time_ms, currents.Ip_A / 1.0e3, color="black", label="$I_p$")
            axis.plot(time_ms, currents.I_cs_A / 1.0e3, label="CS")
            axis.plot(time_ms, currents.I_rf_A / 1.0e3, label="F-P")
            axis.plot(time_ms, currents.I_tail_A / 1.0e3, label="hot tail")
            axis.plot(time_ms, currents.I_vertical_A / 1.0e3, label="vertical charge")
            axis.set_title(case.name, fontsize=10)
            axis.set_xlabel("time [ms]")
            axis.set_ylabel("causal contribution [kA]")
            axis.grid(True, alpha=0.3)
        axes[0, 0].legend(frameon=False, fontsize=7, ncol=3)
        fig.suptitle("Mechanism-resolved L-R response", fontsize=15)
        pdf.savefig(fig)
        plt.close(fig)

        for case in [item for item in cases if item.source_label == "50 GHz X" ]:
            profiles = reconstruct_current_profiles(case.result, case.config)
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
            _draw_profile_page(fig, axes, profiles, case.result, case.config.outputs.profile_snapshot_count)
            fig.suptitle(f"50 GHz X-mode current profile | {case.cs_mode}", fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)

        for payload in threshold_payloads:
            _report_threshold_scan_pages(pdf, payload)

        if base_config.experimental_startup_prior.enabled:
            _write_experimental_prior_page(pdf, base_config, threshold_payloads)

        metadata = pdf.infodict()
        metadata["Title"] = "EXL-50U 0-100 ms EC startup model report"
        metadata["Subject"] = "28/50 GHz, CS/no-CS, F-P feedback and adaptive threshold brackets"
        metadata["Keywords"] = "EXL-50U, EC startup, Fokker-Planck, pre-CFS, threshold"


def run_exl50u_suite(
    base: SimulationConfig,
    output_directory: str | Path,
) -> dict:
    """Run six 0--100 ms cases and adaptive scans for each EC source."""

    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("EC28 O | no CS", "28 GHz O", ("EC28_O",), "without_cs"),
        ("EC28 O | CS", "28 GHz O", ("EC28_O",), "with_cs"),
        ("EC50 X | no CS", "50 GHz X", ("EC50_X",), "without_cs"),
        ("EC50 X | CS", "50 GHz X", ("EC50_X",), "with_cs"),
        ("EC28+50 | no CS", "28+50 GHz", ("EC28_O", "EC50_X"), "without_cs"),
        ("EC28+50 | CS", "28+50 GHz", ("EC28_O", "EC50_X"), "with_cs"),
    )
    cases: list[EXLCase] = []
    for name, source_label, sources, cs_mode in definitions:
        config = _source_case_config(base, sources, cs_mode)
        result = integrate(config)
        slug = name.lower().replace(" ", "_").replace("|", "").replace("+", "plus")
        write_run(result, config, out / "cases" / slug)
        cases.append(EXLCase(name, source_label, cs_mode, config, result))

    threshold_payloads = []
    for source_name in ("EC28_O", "EC50_X"):
        payload = run_threshold_scan(_threshold_config(base, source_name))
        threshold_payloads.append(payload)
        (out / f"threshold_{source_name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    stage_summaries = [
        {
            "case": case.name,
            "source": case.source_label,
            "cs_mode": case.cs_mode,
            "windows": summarize_stage_mechanisms(case.result, case.config),
        }
        for case in cases
    ]
    stage_path = out / "stage_mechanisms.json"
    stage_path.write_text(json.dumps(stage_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    pdf_path = out / "exl50u_0_100ms_formal_report.pdf"
    _write_suite_pdf(cases, threshold_payloads, stage_summaries, base, pdf_path)
    quality_gate = assess_device_prediction_quality(base).to_dict()
    manifest = {
        "report": pdf_path.name,
        "time_window_s": [base.numerics.t_start_s, base.numerics.t_end_s],
        "cases": [_case_summary(case) for case in cases],
        "threshold_files": ["threshold_EC28_O.json", "threshold_EC50_X.json"],
        "stage_mechanisms_file": stage_path.name,
        "experimental_startup_prior": base.experimental_startup_prior.model_dump(mode="json"),
        "prediction_quality_gate": quality_gate,
        "model_warning": (
            "Synthetic Gaussian D_ql and analytic pre-CFS magnetic/loss providers are not "
            "quantitative EXL-50U predictions. No GS/EFIT surfaces are used in 0--100 ms."
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
