from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecstartup.config import AtomicSpeciesConfig, SimulationConfig
from ecstartup.physics.adas import OpenADASAtomicRateProvider, load_openadas_manifest, read_adf11


MANIFEST = "data/openadas/adf11_h_96/manifest.json"
pytestmark = pytest.mark.skipif(
    not Path("data/openadas/adf11_h_96/scd96_h.dat").is_file(),
    reason="optional OpenADAS files are not redistributed; see data/openadas/adf11_h_96/README.md",
)


def _hydrogen() -> AtomicSpeciesConfig:
    return AtomicSpeciesConfig(
        symbol="H",
        atomic_number=1,
        role="fuel",
        max_charge=1,
    )


def test_native_openadas_adf11_grid_and_exact_node_interpolation():
    table = read_adf11("data/openadas/adf11_h_96/scd96_h.dat")
    assert table.nuclear_charge == 1
    assert table.log10_coefficient_cgs.shape == (1, 29, 24)
    i_t, i_n = 8, 12
    coefficient = table.coefficient_si(
        10.0 ** table.log10_Te_eV[i_t],
        1.0e6 * 10.0 ** table.log10_ne_cm3[i_n],
    )
    assert coefficient == pytest.approx(1.0e-6 * 10.0 ** table.log10_coefficient_cgs[0, i_t, i_n])


def test_openadas_manifest_hashes_and_hydrogen_provider_are_physical():
    tables, payload = load_openadas_manifest(MANIFEST)
    assert set(tables) == {"scd", "acd", "plt", "prb"}
    assert payload["database"] == "OpenADAS"
    provider = OpenADASAtomicRateProvider(MANIFEST)
    species = _hydrogen()
    ionization = provider.ionization_rate_m3_s(species, 0, 20.0, 1.0e19)
    recombination = provider.recombination_rate_m3_s(species, 1, 20.0, 1.0e19)
    radiation = provider.radiation_coefficient_W_m3(species, 0, 20.0, 1.0e19)
    assert ionization > 0.0
    assert recombination > 0.0
    assert radiation > 0.0
    assert provider.label.startswith("OpenADAS:")


def test_openadas_rejects_extrapolation_and_can_explicitly_clip():
    provider = OpenADASAtomicRateProvider(MANIFEST)
    species = _hydrogen()
    with pytest.raises(ValueError, match="outside"):
        provider.ionization_rate_m3_s(species, 0, 1.0e8, 1.0e19)
    clipped = OpenADASAtomicRateProvider(MANIFEST, bounds="clip")
    assert np.isfinite(clipped.ionization_rate_m3_s(species, 0, 1.0e8, 1.0e19))


def test_openadas_backend_is_constructed_by_global_model():
    from ecstartup.physics.global_model import GlobalStartupModel

    config = SimulationConfig(
        species_wall={
            "enabled": True,
            "atomic_backend": "openadas_adf11",
            "atomic_manifest": MANIFEST,
            "atomic_bounds": "clip",
            "species": [
                {"symbol": "D", "role": "fuel", "atomic_number": 1, "max_charge": 1}
            ],
        }
    )
    model = GlobalStartupModel(config)
    assert model.species_wall_model.atomic_rates.label.startswith("OpenADAS:")
