# EXL-50U experimental startup pipeline

This pipeline binds shot-resolved diagnostic/coil waveforms to the startup
model without inferring a pre-CFS Grad-Shafranov equilibrium.

## Reproducible entry points

```powershell
python -m ecstartup inspect-exl-shots exl50u_startUpShots `
  --calibration configs/exl50u_shot_calibration.json `
  --output runs/exl50u_startup_shots/shot_audit.json

python -m ecstartup build-exl-validation exl50u_startUpShots `
  --calibration configs/exl50u_shot_calibration.json `
  --output output/pdf/exl50u_validation

python -m ecstartup audit-device-geometry `
  configs/exl50u_device_geometry.json `
  --shot-csv exl50u_startUpShots/22134.csv `
  --output runs/exl50u_startup_shots/geometry_audit_22134.json
```

Each diagnostic retains its native time coordinate. Downsampling keeps local
minima and maxima so short H-alpha/HXR peaks are not discarded. A physical
value is emitted only when the calibration file supplies a conversion;
otherwise plots and JSON say `raw`, `relative_only`, or `unknown`.

## Current validation candidates

The waveform heuristic identifies 22121, 22122, 22128, 22135, and 22136 as
startup-success candidates. Shot 22134 is a termination candidate: its
current peaks near 82 ms and the strongest H-alpha peak is close in time.
These are not authoritative experiment labels. A shot log must confirm the
classification, timing reference, gas species, gyrotron state, and any
diagnostic saturation.

The supplied CSVs contain `Ip`, HCN `ne`, H-alpha, HXR, loop voltage, `BT`,
CS, PF1-PF10, gas monitors, and ECRH0-ECRH3 monitor channels. They do not
contain ECE. ECRH monitors, gas pressure, H-alpha, and HXR remain raw or
relative until their transfer functions are supplied.

## Geometry quality gate

The user drawings establish these identities and relative positions:

- PF1/PF2: upper/lower central-column side;
- PF3/PF4: upper/lower inboard;
- PF5/PF6: upper/lower outer crown;
- PF7/PF8: upper/lower outboard;
- PF9/PF10: upper/lower outboard midplane;
- CS: central solenoid, which must be represented by its actual axial
  sections rather than one invented circular loop.

The drawings do not establish metric `R,Z`, turns, polarity, conductor width
or height, nor a measured first-wall/limiter polygon. Consequently
`configs/exl50u_device_geometry.json` is intentionally `schematic_only`, and
`build_experimental_magnetic_field()` refuses to build a physics field from
it. Set `geometry_status` to `metric_confirmed` only after every required
field is traceable to an engineering source.

TPC and VFC are shot-specific magnetic configurations produced by PF-current
combinations. They are not extra coil channels. Their group definitions must
therefore specify the contributing PF circuits and signs before they can be
used as experimental configuration labels.

## Implemented numerical layers

- Metric geometry plus calibrated PF/CS/BT waveforms can build a vacuum
  Biot-Savart field; a polygonal first wall is used for field-line hits.
- The independent 1D-2V solver advances `f(s,u_parallel,u_perp,t)`, neutral
  and ion density along a traced open field line, including absorbing ends,
  local ionization, collisions, electric field, and spatial non-diagonal
  `Dql`.
- Velocity-grid and relativistic-momentum-grid backends are selected
  explicitly. RF power uses the exact relativistic kinetic energy and RF
  momentum uses `p_parallel=m_e*c*u_parallel` on the relativistic grid.
- Spatial `Dql` is checked for positive semidefiniteness, normalized against
  absorbed power on the current distribution, and rejected if its discrete
  particle balance exceeds the configured tolerance.
- A named `L dI/dt + R I = V` network supports explicit coil-plasma mutual
  inductances and measured-coil induced loop voltage.
- Versioned cross-section tables require provenance plus SHA-256 and can be
  integrated over either a Maxwellian or the instantaneous 2V EEDF.

## Not yet device-predictive

The following items remain hard quality gates rather than completed physics:

1. engineering PF/CS geometry, measured wall/limiter coordinates, TPC/VFC
   group definitions, and the coil-plasma mutual-inductance matrix;
2. time-step-level two-way coupling between the 1D-2V state, the global
   particle/energy model, the circuit state, and evolving vacuum/plasma
   fields;
3. a whole-device, mesh-converged hot-plasma full-wave field for the actual
   28/50 GHz port; a local Nedelec/PML conversion-layer solver and the
   verified-external-field `Dql` contract are implemented;
4. versioned LXCat/AMJUEL/HYDHEL molecular data; official checksummed
   OpenADAS H-96 `SCD/ACD/PLT/PRB` tables are implemented for atomic H/D/T;
5. production-grid acceleration and material-specific boundary data for the
   implemented relativistic Braams-Karney/Beliaev-Budker `Cee+Cei` reference
   operator and Debye-scale Poisson sheath;
6. shot-log-confirmed success/failure labels plus calibrated ECRH power, gas
   pressure, density convention, HXR/H-alpha, and ECE when available;
7. EXL-50U port-level validation of the internal local Nedelec/PML X-B/EBW
   solver, harmonic absorption, FLR `Dql` and two-way EBWCD closure. Until
   actual port geometry/profile inputs pass the gates, imported verified
   whole-device fields remain the quantitative route.

Until these gates pass, the output is mechanism and sensitivity analysis. It
must not be described as an accurate prediction of startup success or
`Ip(t)`.
