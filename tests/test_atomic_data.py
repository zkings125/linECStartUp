from __future__ import annotations

import csv
import hashlib
import json
import math

import numpy as np
import pytest

from ecstartup.config import GasConfig
from ecstartup.constants import E_CHARGE_C, M_E_KG
from ecstartup.physics.atomic_data import CrossSectionRateModel, load_normalized_cross_sections
from ecstartup.physics.fokker_planck import ReducedFokkerPlanck2V, VelocityGrid


def _write_dataset(tmp_path, *, corrupt_hash: bool = False):
    processes = []
    energy = np.linspace(0.0, 500.0, 1001)
    for process, sigma in (("ionization", 2.0e-20), ("excitation", 1.0e-20), ("momentum", 3.0e-20)):
        path = tmp_path / f"{process}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["energy_eV", "cross_section_m2"])
            writer.writerows(zip(energy, np.full_like(energy, sigma)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        processes.append({"process": process, "file": path.name, "sha256": "0" * 64 if corrupt_hash else digest})
    manifest = {
        "database": "LXCat-test-snapshot",
        "version": "test-1",
        "retrieved_date": "2026-10-26",
        "species": "D2",
        "source_url": "https://example.invalid",
        "citation": "synthetic unit-test data",
        "license_note": "test only",
        "processes": processes,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_versioned_cross_sections_reproduce_constant_sigma_maxwellian_rate(tmp_path):
    tables = load_normalized_cross_sections(_write_dataset(tmp_path))
    temperature = 10.0
    expected_speed = math.sqrt(8.0 * E_CHARGE_C * temperature / (math.pi * M_E_KG))
    assert tables["ionization"].maxwellian_rate_m3_s(temperature) == pytest.approx(
        2.0e-20 * expected_speed,
        rel=2.0e-3,
    )
    model = CrossSectionRateModel(GasConfig(), tables)
    rates = model.evaluate(temperature, 1.0e17, 1.0e19, 10.0)
    assert rates.K_ion_m3_s > rates.K_exc_m3_s
    assert rates.eedf_label == "LXCat-test-snapshot"


def test_cross_section_rate_can_be_integrated_over_the_actual_eedf(tmp_path):
    tables = load_normalized_cross_sections(_write_dataset(tmp_path))
    grid = VelocityGrid.uniform(81, 48, 5.0e6, 5.0e6)
    solver = ReducedFokkerPlanck2V(grid, cross_section_area_m2=1.0)
    state = solver.maxwellian_state(1.0e17, 10.0)
    rates = CrossSectionRateModel(GasConfig(), tables).eedf_rates(grid, state.f_e)
    assert rates["ionization"] > 0.0
    assert rates["momentum"] == pytest.approx(1.5 * rates["ionization"])


def test_atomic_snapshot_checksum_mismatch_is_fatal(tmp_path):
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_normalized_cross_sections(_write_dataset(tmp_path, corrupt_hash=True))
