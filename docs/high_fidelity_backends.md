# High-fidelity kinetic, sheath, atomic and full-wave backends

This document separates implemented physics from remaining device-validation
requirements. None of these backends uses a pre-CFS Grad-Shafranov surface.

## Nonlinear Landau-Rosenbluth collisions

`NonlinearLandauRosenbluth2V` evaluates the nonlinear nonrelativistic Landau
integral for a gyrotropic `f(v_parallel,v_perp)`. It is equivalent to the
classical Rosenbluth formulation and also exposes direct H/G convolution
potentials. The discrete collision term is projected onto the orthogonal
complement of `1`, `v_parallel`, and `v^2/2`, so particle number, parallel
momentum, and kinetic energy are conserved to solver precision.

The direct quadrature scales as `O(Nv^2 Nphi)` and is a correctness backend,
not the eventual high-resolution production algorithm. The optional
`rosenbluth_operator` argument in `OpenFieldLineKinetic1D2V.advance()` applies
it cell-by-cell through operator splitting. It cannot be enabled together
with the legacy scalar collision frequency.

`RelativisticBraamsKarney2V` supplies the relativistic reference backend. It
evaluates the covariant Beliaev-Budker Landau tensor directly, which is the
integral representation equivalent to the Braams-Karney differential
potentials, and adds the infinite-ion-mass relativistic Lorentz `Cei` term.
`Cee` is projected onto particle, parallel-momentum and relativistic-energy
invariants; `Cei` conserves electron particles and energy while transferring
momentum to the ion background. The direct implementation is
`O(Nu^2 Nphi)` and is intended for verification-sized grids. High-resolution
production still needs the faster coupled-potential multigrid solver.

Reference: Rosenbluth, MacDonald, and Judd, *Physical Review* 107, 1 (1957),
DOI `10.1103/PhysRev.107.1`. Relativistic boundary: Braams and Karney,
*Physical Review Letters* 59, 1817 (1987), DOI
`10.1103/PhysRevLett.59.1817`.

## Spatial ambipolar field and wall sheath

`QuasineutralAmbipolarSheath1D` computes at every kinetic time step:

```text
E_parallel = -(1/e ne) d(ne e Te)/ds
Gamma_i = ni sqrt[e(Te + gamma_i Ti)/m_i]
(1-delta_SEE) Gamma_e0 exp(-Delta_phi/Te) = Gamma_i
```

The interior field is added to the imposed parallel electric field. At each
open end, electrons below `e Delta_phi` are specularly reflected and
super-barrier electrons are absorbed. Diagnostics report both sheath drops,
floating-current residuals, and the spatial quasineutrality residual.

`DebyePoissonAmbipolarSheath1D` extends this closure with separate
Debye-length grids at both material boundaries. It solves nonlinear Poisson's
equation with Boltzmann electrons, cold Bohm ions and the floating potential,
and reports the resolved charge density and Poisson residual. The interior
open-field-line grid remains quasineutral. A material-resolved secondary-
electron energy spectrum remains an external wall input.

## Official OpenADAS ADF11 data

The repository contains an attributed, checksummed manifest for the official
OpenADAS hydrogen-96 ADF11 files: `SCD`, `ACD`, `PLT`, and `PRB`. The data
files are not redistributed because the OPEN-ADAS terms require prior written
permission for bundling them with modelling code. Follow
`data/openadas/adf11_h_96/README.md` to download them for authorized local use.

```powershell
python -m ecstartup inspect-adas data/openadas/adf11_h_96/manifest.json `
  --Te-eV 20 --ne-m3 1e19
```

ADF11 base-10 log grids are interpolated in `(log10 Te, log10 ne)` and
converted from `cm^3/s` or `W cm^3` to SI. Extrapolation is rejected by
default. The `openadas_adf11` species-wall backend currently accepts only
H/D/T charge states 0/1 because this optional dataset is hydrogenic. ADAS
does not replace molecular H2/D2 reactions: AMJUEL/HYDHEL or a versioned
LXCat/BOLSIG+ dataset is still required for dissociation, molecular ions,
vibrational channels, attachment, and momentum transfer.

```yaml
species_wall:
  enabled: true
  atomic_backend: openadas_adf11
  atomic_manifest: data/openadas/adf11_h_96/manifest.json
  atomic_bounds: raise
```

## Verified full-wave input and Dql

`load_fullwave_npz()` accepts a polarization-resolved 2D field map containing
complex `E_plus`, `E_minus`, `E_parallel`, `k_parallel`, `B`, local absorbed
power density, solver identity, convergence state, mesh-verification state,
and the complete port-power budget. Quantitative use is rejected unless

```text
Pincident = Preflected,O + Preflected,X
          + Ptransmitted,O + Ptransmitted,X
          + PEBW,out + Pabsorbed
```

closes within tolerance. The accepted O-X-B efficiency is `PEBW,out/Pincident`.
For the axisymmetric 2D contract, the independent volume integral
`integral 2*pi*R*p_abs(R,Z) dR dZ` must also match the port-budget absorbed
power within the same tolerance; otherwise the input is rejected before any
`Dql` projection.

```powershell
python -m ecstartup inspect-fullwave path/to/fullwave_result.npz
```

The input NPZ contract is:

```text
R_m[nR], Z_m[nZ]
E_plus_real_V_m[nR,nZ], E_plus_imag_V_m[nR,nZ]
E_minus_real_V_m[nR,nZ], E_minus_imag_V_m[nR,nZ]
E_parallel_real_V_m[nR,nZ], E_parallel_imag_V_m[nR,nZ]
k_parallel_m_inv[nR,nZ], B_T[nR,nZ]
absorbed_power_density_W_m3[nR,nZ]
metadata_json scalar UTF-8 JSON
```

`metadata_json` must contain `frequency_Hz`, `solver_name`,
`solver_version`, `converged`, `mesh_verified`, and a `power_budget` object
with `incident_W`, reflected/transmitted O/X powers, `ebw_out_W`, and
`absorbed_W`. Boolean declarations are an import gate, not independent proof;
the originating solver's convergence and mesh-study artifacts must remain
part of the reproducibility record.

Mechanism identification additionally requires `wave_branch` equal to
`electromagnetic_ec`, `ebw`, or `mixed`, plus
`mode_decomposition_verified=true`. This declaration must be backed by the
originating solver's eigenmode projection or resolved `(k_parallel,k_perp)`
spectrum. A positive `ebw_out_W` without that decomposition passes neither the
EBW identity gate nor the quantitative EBWCD gate; linEC reports it as
`unresolved_fullwave`.

`project_fullwave_fundamental_dql()` interpolates the verified complex field
onto an `OpenFieldLineGrid` and evaluates the relativistic fundamental
resonance `omega-k_parallel*v_parallel-Omega_e/gamma=0`. The diffusion tensor
is an outer product and is therefore positive semidefinite. Its absolute
power is normalized by the existing 1D-2V EEDF response.

This projection covers the fundamental resonance. Second-harmonic FLR
coupling and hot EBW harmonic absorption require a validated
Kennel-Engelmann/adjoint implementation.

To produce a file directly consumable by the spatial kinetic workflow, give
the command an open-field-line NPZ containing equal-length `s_m`, `R_m`,
`Z_m`, `B_T`, and `cross_section_area_m2` arrays:

```powershell
python -m ecstartup project-fullwave-dql fullwave_result.npz open_fieldline.npz `
  configs/linec_parameters.ini --output runs/fullwave/spatial_dql.npz `
  --summary runs/fullwave/summary.json
```

For an EBW/mixed branch or second-harmonic projection, the resolved full-wave
spectrum must also supply a defensible `k_perpendicular`; pass its selected
value explicitly so the code cannot silently invent the FLR scale:

```powershell
python -m ecstartup project-fullwave-dql fullwave_result.npz open_fieldline.npz `
  configs/linec_parameters.ini --harmonic 2 --k-perpendicular-m-inv 1200 `
  --output runs/fullwave/spatial_ebw_dql.npz
```

The output preserves all spatial coordinates, the velocity or relativistic
momentum grid, all three diffusion components, solver provenance, port-power
budget, and the explicit statement that no pre-CFS equilibrium was used.
`load_spatial_fullwave_dql_npz()` reloads it as a
`SpatialQuasilinearDiffusion` and rejects any spatial/kinetic grid mismatch
before `OpenFieldLineKinetic1D2V.advance()` is called.

`solve_hot_plasma_nedelec()` is an internal local 2D H(curl) solver using
first-order Nedelec elements and a complex-coordinate rectangular PML. Its
dielectric combines the Stix tensor with Doppler-broadened `n=1/n=2`
finite-Larmor-radius loss channels. `solve_hot_plasma_mesh_convergence()`
requires an explicit two-grid absorbed-power comparison; the result also
reports algebraic source/plasma/PML power balance and cells per wavelength.

This is a local Cartesian X-B/EBW conversion-layer solver, not a whole-device
antenna/port model. A device claim still requires actual port geometry,
polarization, density/temperature maps, wall shape and a converged mesh/PML
study. `project_fullwave_harmonic_dql()` supplies the `J_(n-1)(k_perp rho)^2`
FLR coupling and relativistic `n=1/n=2` resonance. The EBW current closure
combines O-X efficiency, local X-B/absorption power, wave momentum and an F-P
current check; a failed mesh, power or current-consistency gate leaves the
result non-quantitative.
