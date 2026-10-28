# API 与模块契约

## 用户入口

```python
from ecstartup import load_config, run_simulation

config = load_config("configs/minimal.yaml")
result = run_simulation(config, output_directory="runs/example")
print(result.final_state.Te_eV)
```

完整积分也可以复用已装配的模型：

```python
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.physics.magnetic import make_magnetic_field

field = make_magnetic_field(config.geometry)
model = GlobalStartupModel(config, magnetic_field=field)
result = run_simulation(config, output_directory="runs/example", model=model)
```

`model` 省略时，`run_simulation` 仍按配置选择默认 provider。

命令行等价于：

```text
python -m ecstartup validate <config>
python -m ecstartup run <config> [--output <dir>]
python -m ecstartup trace <config> [--R <m>] [--Z <m>] [--direction forward|backward|both]
python -m ecstartup raytrace-ec <config> [--source <name>] [--time <s>] [--Ip <A>] [--ne <m^-3>]
python -m ecstartup analyze-oxb <config> [--source <name>] [--time <s>] [--Ip <A>] [--average-ne <m^-3> | --peak-ne <m^-3>]
python -m ecstartup solve-fp <config> [--output <summary.json>] [--distribution-output <distribution.npz>]
python -m ecstartup assess-dql <config> [--dql-file <Dql.npz>] [--output <assessment.json>]
python -m ecstartup threshold-scan <config> [--output <scan.json>] [--pdf-output <scan.pdf>]
python -m ecstartup scan <config> [--Ip <A>] [--time <s>] [--radial-seeds N] [--poloidal-seeds M]
    [--step <m>] [--max-length <m>] [--invalid-policy record|raise] [--output <json>]
python -m ecstartup compare <matrix.json> [--output <json>]
python -m ecstartup ensemble <config> --parameters <ranges.json> [--samples N] [--output <json>]
python -m ecstartup ablate <config> [--output <json>]
python -m ecstartup postprocess <run_dir>
```

`compare` 的位置参数是包含 `base_config` 和 `devices` 的比较矩阵；`run`、
`ensemble` 和 `ablate` 的位置参数都必须是一个完整的单案例配置。特别是，
`ablate configs/tpc_vfc_matrix.json` 不合法，因为矩阵不是
`SimulationConfig`；要对矩阵案例消融，请先用
`comparison.load_device_matrix()` 载入各案例，再对每个配置调用
`ablation.run_ablation()`（示例见 [`docs/tpc_vfc.md`](tpc_vfc.md)）。

`trace` 只做固定时刻的轴对称场线诊断，不会运行或修改 0D 状态。起点默认
使用 RF 发射点（未设置时为磁轴），也可用 `--R`、`--Z`、`--Ip` 和 `--time`
覆盖；`--no-path` 可省略路径数组以减少 JSON 大小。详细字段和收敛要求见
[`docs/fieldline.md`](fieldline.md)。

`raytrace-ec` 使用冷等离子体 Stix 色散行列式积分轴对称 Hamiltonian 射线。
它只需要瞬时 `MagneticFieldProvider`，不需要 GS/EFIT 磁面。输出可传播性、
谐波共振、WKB 指标和完整路径；当前不返回热等离子体吸收。详见
[`docs/ec_ray_tracing.md`](ec_ray_tracing.md)。

`analyze-oxb` 使用空间密度 provider 和同一瞬时磁场查找 O cutoff/UHR，输出
冷 O-X 代理和 EBW 到达谐波层的几何候选。它不返回 X-B/联合效率，也不把
候选标记当作 EBWCD 电流预测。

`solve-fp` 运行独立 `ReducedFokkerPlanck2V`，由
`J_parallel=q_e*integral(v_parallel*f_e*d3v)` 计算电流矩。它不调用 GS/EFIT，
也不自动把结果反馈给 `run` 的电路方程。外部波程序可通过 NPZ 注入对角
`D_ql`；接口和限制见 [`fokker_planck.md`](fokker_planck.md)。

`assess-dql` 不运行 F-P，而是检查当前 `D_ql` 的来源、坐标基、是否依赖闭合
磁面、功率归一化、波分支以及 O-X-B 转换证据。配置中写
`launch_mode='O-X-B'` 不会被当成 EBW 已生成；外部全波文件还必须给出经过
验证的模式/波数分支分解，才能把电流机制标记为 EBWCD。

`threshold-scan` 扫描配置中的压强、`Z_eff`、EC 功率和 CS 模式。每个点与
同条件零功率基线比较，输出 EC 增量电流和离散阈值。详见
[`threshold_scan.md`](threshold_scan.md)。

`scan` 批量追踪极向截面种子并输出面积加权的场线拓扑统计。`--Ip` 未指定时
使用配置中的 `initial.Ip_A`；`--radial-seeds`/`--poloidal-seeds` 分别控制
径向环带和极向角数量，`--step`/`--max-length` 控制追踪积分，
`--invalid-policy record|raise` 控制线圈奇异点和无效轨迹处理。默认 JSON 顶层
就是扫描结果；增加 `--topology` 可附带兼容现有 `TopologyResult` 的 `topology`
对象。`scan` 不会改变 0D 模型，详见 [`docs/fieldline.md`](fieldline.md)。

## 核心对象

### `ecstartup.state.GlobalState`

```python
GlobalState(
    ne_m3: float,
    ni_m3: float,
    nn_m3: float,
    electron_energy_J_m3: float,
    Ip_A: float,
    n_hot_m3: float = 0.0,
    hot_energy_J_m3: float = 0.0,
)
```

`to_vector()` / `from_vector()` 用于积分器；`Te_eV` 和 `hot_mean_energy_eV` 是派生量；`projected()` 只做数值正性保护。

### `ecstartup.physics.global_model.GlobalStartupModel`

```python
model = GlobalStartupModel(config)
state0 = model.initial_state()
dy = model.rhs(t_s, state0.to_vector())
evaluation = model.evaluate(t_s, state0)
```

`evaluation.values` 是无文件副作用的标量诊断字典，`evaluation.rf/circuit/topology` 分别保存结构化源项和拓扑结果。

启用 `config.species_wall.enabled` 后，模型自动构造 `DYONSpeciesWallModel`，状态由固定 7 分量扩展为 `ExtendedGlobalState`。自定义 ADAS 后端通过模块注入：

```python
from ecstartup.physics.species_wall import DYONSpeciesWallModel

species_wall = DYONSpeciesWallModel(
    config.species_wall,
    config.geometry,
    atomic_rates=my_adas_provider,
)
model = GlobalStartupModel(config, species_wall_model=species_wall)
```

`integrate` 会从模型读取动态状态布局与逐分量绝对容差；调用方不应假设状态向量恒为 7 分量。

要替换解析磁场而不改 ODE，可通过关键字参数注入 provider：

```python
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.physics.magnetic import make_magnetic_field

field = make_magnetic_field(config.geometry)  # 或真实线圈/场线 provider
model = GlobalStartupModel(config, magnetic_field=field)
```

`current_drive=` 和 `circuit_model=` 也支持同样的构造级替换；未显式传入时
仍使用当前的解析实现。provider 只负责物理量，不负责读取配置文件或写输出。

### `ecstartup.physics.current_drive.CurrentDriveModel`

```python
components = CurrentDriveModel(config.geometry, config.rf).evaluate(
    state, rf_terms, magnetic_state=field_state, loss_times=loss_times
)
```

它产生 `I_RF`、`I_tail`（兼容别名 `I_orbit`/`I_tpc`）和
`I_vertical_charge`（兼容别名 `I_vfc`/`I_vtf`）；欧姆电流、电阻和电感仍由
`CircuitModel` 负责。这里的 `I_*` 是体积分电流（A），不能与 EXL-50 论文
中以 `j_TPC`、`j_VTF` 表示的探针/局部电流密度记号直接数值相等。TPC/VFC
是磁场 provider 的配置名，不是这里的电流字段名：`TPC` 不会自动等于
`I_tpc`，`VFC` 也不会自动等于 `I_vfc`。

### `ecstartup.physics.magnetic`

```python
from ecstartup.physics.magnetic import make_magnetic_field

field = make_magnetic_field(config.geometry)
point = field.evaluate(config.geometry.R0_m, 0.0, state.Ip_A, t_s)
field_state = field.configuration_state(state.Ip_A, t_s)
```

`field_configuration="TPC"` 选择镜像/捕获粒子 surrogate，
`field_configuration="VFC"` 选择垂直场 surrogate；两者都实现同一个
`MagneticFieldProvider` 契约，便于替换真实线圈场、GS 和场线追踪。

若 `geometry.coil_components` 非空，工厂返回
`CompositeMagneticField`，将基础 provider 与各个独立的静态/时变线圈
分量叠加：

```python
field = make_magnetic_field(config.geometry)
point = field.evaluate(R_m, Z_m, Ip_A, t_s)
```

每个分量用 `B_R_T_per_A`、`B_Z_T_per_A`、`B_phi_T_per_A` 和
`current_waveform_A` 描述；`role` 只是 `TPC`/`VFC` 标签，不会自动开启
任何电流机制。解析分量的适用边界和真实 provider 注入方式见
[`docs/coil_superposition.md`](coil_superposition.md)。

对于需要实际圆环线圈场的第一步升级，可注入
`ecstartup.physics.biot_savart.BiotSavartMagneticField`。它支持独立时变
电流、有限多匝离散线圈和椭圆积分场值；用法及理想细丝奇异点约定见
[`docs/biot_savart.md`](biot_savart.md)。该 provider 不自动替代 GS/CFS
拓扑判据。

### `ecstartup.physics.fieldline.FieldLineTracer`

`FieldLineTracer` 独立于 0D ODE，沿任意 `MagneticFieldProvider` 的
`(B_R,B_Z,B_phi)` 以弧长积分场线，支持 `forward`、`backward` 和 `both`
方向，并返回结构化的 `FieldLineTrace`（连接长度、壁命中/闭合/超时状态、
端点、`min_B`/`max_B` 和镜像比）。矩形边界可由 `GeometryConfig` 或
`FieldLineBounds` 提供；完整示例和收敛注意事项见
[`docs/fieldline.md`](fieldline.md)。

### `ecstartup.physics.tabulated_field.TabulatedMagneticField`

`TabulatedMagneticField` 将规则 `(R,Z)` 网格上的 `B_R/B_Z/B_phi`（以及可选
`psi`）接入同一个 `MagneticFieldProvider` 契约。它是静态场图：`evaluate`
保留 `Ip_A` 与 `t_s` 参数但不会自动缩放或插值时间。边界策略必须显式选择
`raise`、`clip`、`extrapolate` 或 `nan`；构造时还会记录 `divB` 与 `psi`
导数残差。场图不会自动推断 TPC/VFC，镜像比、捕获比例和连接长度因子只能
作为显式元数据传入。NPZ/xarray 加载和示例见
[`docs/tabulated_field.md`](tabulated_field.md)。

### `ecstartup.physics.fieldline_topology.FieldLineTopologyScanner`

`FieldLineTopologyScanner` 在极向截面生成径向/极向种子，调用独立的
`FieldLineTracer`，并按环带面积权重返回每条场线状态和汇总统计。结果中的
`closed_fraction` 是有效场线中的闭合权重比例，`closed_area_fraction` 是总
种子面积中的闭合权重比例；无效场线不会被静默计入分母。命令行入口为
`python -m ecstartup scan CONFIG`，完整字段和收敛限制见
[`docs/fieldline.md`](fieldline.md)。

### `ecstartup.physics.fokker_planck.ReducedFokkerPlanck2V`

```python
from ecstartup.physics.fokker_planck import ReducedFokkerPlanck2V, VelocityGrid

grid = VelocityGrid.uniform(81, 64, 4.0e6, 4.0e6)
solver = ReducedFokkerPlanck2V(grid, cross_section_area_m2=0.20)
state = solver.maxwellian_state(1.0e17, 20.0)
state, diagnostics = solver.advance(
    state,
    1.0e-8,
    E_parallel_V_m=1.0,
    collision_frequency_s=2.0e5,
    collision_temperature_eV=20.0,
)
current_A = solver.moments(state).current_parallel_A
```

`QuasilinearDiffusion2V` 接收与网格同形的非负对角扩散数组；
`KineticRunResult` 负责 CLI 摘要和 NPZ 输出。完整 CQL3D 模块映射及
pre-CFS 适用边界见 [`fokker_planck.md`](fokker_planck.md)。

### `ecstartup.physics.open_field_loss.OpenFieldLossModel`

```python
loss = OpenFieldLossModel(config.geometry, rates, config.initial.Ti_eV)
times = loss.evaluate(state, closed_fraction=0.0)
```

该模块隔离 `L_conn/v_th`、Bohm 声速和 CFS 候选后的损失时间反馈；在完整
工作流中应把拓扑 provider 返回的 `connection_length_m` 显式传入，这样
TPC/VFC 的连接长度因子只应用一次。接入真实场线追踪/鞘层模型时只替换此
provider。

### 可替换 provider

```python
class RateProvider(Protocol):
    def evaluate(self, Te_eV, ne_m3, nn_m3, EoverN_Td): ...

class RFProvider(Protocol):
    def evaluate(self, t_s, state): ...

class TopologyProvider(Protocol):
    def detect(self, Ip_A, t_s=0.0): ...

class MagneticFieldProvider(Protocol):
    def evaluate(self, R_m, Z_m, Ip_A, t_s=0.0): ...
    def flux_function(self, R_m, Z_m, Ip_A, t_s=0.0): ...
    def configuration_state(self, Ip_A, t_s=0.0): ...
```

生产实现只需满足 `ecstartup/contracts.py` 的接口，并在 workflow 装配时注入；不应从 provider 内部读取 YAML 或写 NetCDF。

## 扩展接口（尚未实现）

`ecstartup.physics.kinetic.NotImplementedFokkerPlanck` 和 `physics.equilibrium.NotImplementedGradShafranov` 是有意保留的显式占位。它们调用时会抛出 `NotImplementedError`，避免用户误以为圆截面替代器已经具备 2V/GS 物理。
