"""可替换的物理闭合模块。"""

from .circuit import CircuitModel
from .charge_closure import ChargeClosureModel, ChargeClosureTerms
from .current_drive import CurrentDriveModel
from .biot_savart import (
    BiotSavartMagneticField,
    CircularLoop,
    circular_loop_field,
    circular_loop_flux,
    make_biot_savart_field,
)
from .fieldline import (
    AxisymmetricFieldLineTracer,
    FieldLineBounds,
    FieldLineBranch,
    FieldLineTrace,
    FieldLineTracer,
    trace_field_line,
)
from .fieldline_topology import (
    FieldLineScanConfig,
    FieldLineScanSample,
    FieldLineTopologyDetector,
    FieldLineTopologyError,
    FieldLineTopologySample,
    FieldLineTopologyScanAdapter,
    FieldLineTopologyScanResult,
    FieldLineTopologyScanner,
    FieldLineScanTopologyDetector,
    FieldLineTopologyScanConfig,
    FieldLineScanResult,
    FieldLineTopologyAdapter,
)
from .magnetic import (
    AnalyticCoilComponent,
    CircularMagneticField,
    CompositeMagneticField,
    MagneticConfigurationState,
    MagneticFieldProvider,
    TPCMagneticField,
    VFCMagneticField,
    make_magnetic_field,
)
from .open_field_loss import OpenFieldLossModel
from .rates import CollisionRateModel
from .rf import ECRHSource
from .topology import SurrogateTopologyDetector
from .species_wall import (
    AnalyticAtomicRateProvider,
    AtomicRateProvider,
    DYONSpeciesWallModel,
    ExtendedGlobalState,
    SpeciesWallLayout,
    SpeciesWallState,
    SpeciesWallTerms,
)
from .tabulated_field import (
    TabulatedFieldProvider,
    TabulatedMagneticField,
    TabulatedMagneticFieldProvider,
)
from .eqdsk import GFileData, read_gfile
from .mat_equilibrium import MatEquilibriumReference, read_mat_equilibrium
from .fokker_planck import RelativisticMomentumGrid, RelativisticFokkerPlanck2V
from .orbit_average import OrbitClassification, classify_pitch, bounce_average, apply_bounce_weight_to_diffusion
from .spatial_fokker_planck import (
    OpenFieldLineGrid,
    OpenFieldLineKinetic1D2V,
    SpatialKineticDiagnostics,
    SpatialKineticState,
    SpatialQuasilinearDiffusion,
)
from .inductive_network import (
    CircuitNetworkDiagnostics,
    CircuitNetworkState,
    InductiveCircuitNetwork,
)
from .atomic_data import (
    AtomicDataProvenance,
    CrossSectionRateModel,
    CrossSectionTable,
    load_normalized_cross_sections,
)
from .adas import ADF11Table, OpenADASAtomicRateProvider, load_openadas_manifest, read_adf11
from .wall_sheath import QuasineutralAmbipolarSheath1D, SpatialSheathSolution
from .rosenbluth import (
    NonlinearLandauRosenbluth2V,
    RosenbluthDiagnostics,
    RosenbluthPotentials2V,
)
from .relativistic_collisions import (
    RelativisticBraamsKarney2V,
    RelativisticCollisionDiagnostics,
)
from .poisson_sheath import (
    DebyePoissonAmbipolarSheath1D,
    DebyeSheathProfile,
    PoissonSpatialSheathSolution,
)
from .hot_fullwave import (
    HotFullWaveResult,
    HotPlasmaDielectric,
    NedelecPMLConfig,
    solve_hot_plasma_mesh_convergence,
    solve_hot_plasma_nedelec,
)
from .fullwave_dql import (
    FullWaveDqlProjection,
    FullWaveFieldMap2D,
    FullWavePowerBudget,
    load_fullwave_npz,
    load_open_fieldline_npz,
    load_spatial_fullwave_dql_npz,
    project_fullwave_fundamental_dql,
    project_fullwave_harmonic_dql,
    save_fullwave_dql_npz,
)
from .ebw_current_drive import EBWCurrentDriveClosure, close_ebw_current_drive

__all__ = [
    "CircuitModel",
    "ChargeClosureModel",
    "ChargeClosureTerms",
    "CurrentDriveModel",
    "CircularLoop",
    "BiotSavartMagneticField",
    "circular_loop_field",
    "circular_loop_flux",
    "make_biot_savart_field",
    "AxisymmetricFieldLineTracer",
    "FieldLineBounds",
    "FieldLineBranch",
    "FieldLineTrace",
    "FieldLineTracer",
    "trace_field_line",
    "FieldLineScanConfig",
    "FieldLineScanSample",
    "FieldLineTopologyDetector",
    "FieldLineTopologyError",
    "FieldLineTopologySample",
    "FieldLineTopologyScanAdapter",
    "FieldLineTopologyScanResult",
    "FieldLineTopologyScanner",
    "FieldLineScanTopologyDetector",
    "FieldLineTopologyScanConfig",
    "FieldLineScanResult",
    "FieldLineTopologyAdapter",
    "AnalyticCoilComponent",
    "CircularMagneticField",
    "CompositeMagneticField",
    "TPCMagneticField",
    "VFCMagneticField",
    "MagneticFieldProvider",
    "MagneticConfigurationState",
    "make_magnetic_field",
    "OpenFieldLossModel",
    "CollisionRateModel",
    "ECRHSource",
    "SurrogateTopologyDetector",
    "AtomicRateProvider",
    "AnalyticAtomicRateProvider",
    "DYONSpeciesWallModel",
    "ExtendedGlobalState",
    "SpeciesWallLayout",
    "SpeciesWallState",
    "SpeciesWallTerms",
    "TabulatedMagneticField",
    "TabulatedFieldProvider",
    "TabulatedMagneticFieldProvider",
    "GFileData",
    "read_gfile",
    "MatEquilibriumReference",
    "read_mat_equilibrium",
    "RelativisticMomentumGrid",
    "RelativisticFokkerPlanck2V",
    "OrbitClassification",
    "classify_pitch",
    "bounce_average",
    "apply_bounce_weight_to_diffusion",
    "OpenFieldLineGrid",
    "OpenFieldLineKinetic1D2V",
    "SpatialKineticDiagnostics",
    "SpatialKineticState",
    "SpatialQuasilinearDiffusion",
    "CircuitNetworkDiagnostics",
    "CircuitNetworkState",
    "InductiveCircuitNetwork",
    "AtomicDataProvenance",
    "CrossSectionRateModel",
    "CrossSectionTable",
    "load_normalized_cross_sections",
    "ADF11Table",
    "OpenADASAtomicRateProvider",
    "load_openadas_manifest",
    "read_adf11",
    "QuasineutralAmbipolarSheath1D",
    "SpatialSheathSolution",
    "NonlinearLandauRosenbluth2V",
    "RosenbluthDiagnostics",
    "RosenbluthPotentials2V",
    "RelativisticBraamsKarney2V",
    "RelativisticCollisionDiagnostics",
    "DebyePoissonAmbipolarSheath1D",
    "DebyeSheathProfile",
    "PoissonSpatialSheathSolution",
    "HotFullWaveResult",
    "HotPlasmaDielectric",
    "NedelecPMLConfig",
    "solve_hot_plasma_mesh_convergence",
    "solve_hot_plasma_nedelec",
    "FullWaveDqlProjection",
    "FullWaveFieldMap2D",
    "FullWavePowerBudget",
    "load_fullwave_npz",
    "load_open_fieldline_npz",
    "load_spatial_fullwave_dql_npz",
    "project_fullwave_fundamental_dql",
    "project_fullwave_harmonic_dql",
    "save_fullwave_dql_npz",
    "EBWCurrentDriveClosure",
    "close_ebw_current_drive",
]
