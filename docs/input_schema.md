# 输入参数说明（schema_version 1.0）

配置顶层是严格对象，未知字段会被拒绝。所有数值字段均为 SI，单位写在字段名中；温度 `Te_eV/Ti_eV` 使用 eV 是有意的，能量密度仍使用 J/m³。

除 YAML/JSON 外，`load_config` 和所有 CLI 命令也接受带注释的 `.ini/.cfg` 参数文件。推荐用户入口及语法见 [`user_parameters.md`](user_parameters.md)，完整模板见 [`configs/linec_parameters.ini`](../configs/linec_parameters.ini)。

## 顶层

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `schema_version` | string=`"1.0"` | 输入格式版本 |
| `model` | string=`global0d_surrogate` | 当前唯一可运行后端 |
| `geometry` | object | 几何、磁场和连接长度 |
| `gas` | object | 气体与反应率占位参数 |
| `initial` | object | 初始粒子、能量、电流 |
| `rf` | object | ECRH/RF 源 |
| `ray_tracing` | object | 可选冷等离子体 Hamiltonian EC 射线诊断参数 |
| `ebw` | object | O-X-B 可用性与示意 EBW 路径参数 |
| `circuit` | object | 回路电压和 L-R 参数 |
| `thresholds` | object | 阶段事件判据 |
| `numerics` | object | 时间积分参数 |
| `outputs` | object | 输出路径和频率 |
| `species_wall` | object | 可选 DYON-like 逐电荷态与壁模块，默认关闭 |
| `charge_closure` | object | 电子/离子电荷连续性和双极势闭合，默认 `legacy` |
| `seed` | integer=0 | 为未来 Monte Carlo 保留 |

## `geometry`

`R0_m`（大半径）、`minor_radius_m`（小半径）、`volume_m3`、`wall_area_m2`、`connection_length_m`、`B_t_T`、`B_vertical_T`、`B_penetration_T`、`closed_confinement_gain`、可选 `vertical_field_waveform=[[time_s,value_T],...]`、`R_bounds_m=[Rmin,Rmax]`、`Z_bounds_m=[Zmin,Zmax]`。体积和壁面积缺省时按圆截面环形几何计算。`closed_confinement_gain=0` 时 CFS 只作诊断；大于零时用候选闭合比例延长损失时间，是显式的阶段反馈替代项。波形时间必须严格递增。

### `geometry.field_configuration`

`field_configuration` 选择磁场位形 provider：

| 值 | 含义 | 当前 v0.1 surrogate 的作用 |
|---|---|---|
| `circular` | 圆截面基线 | 保持旧版自场/垂直场公式 |
| `TPC` | trapped-particle configuration，捕获粒子/镜像型位形 | 使用 `tpc_mirror_ratio`、`tpc_trapped_fraction` 和 `tpc_decay_index` 修改有效开场线约束，并在诊断中报告热尾约束指标 |
| `VFC` | vertical-field configuration，垂直场位形 | 使用 `field_configuration_scale` 缩放外加垂直场，并使用 `vfc_decay_index`（径向剖面指数）和 `vfc_connection_length_factor` 表示解析位形闭合 |

`field_configuration` 是基础 provider 选择器：每个运行仍只能选
`circular`、`TPC` 或 `VFC` 之一；通过下方 `coil_components` 可以在其上表达
TPC/VFC 线圈的时变叠加。`field_configuration_scale` 只是解析 provider 的幅值校准，不能替代
线圈电流；TPC 中它校准等效自场，VFC 中它缩放外加垂直场。垂直场的时间
变化应通过 `vertical_field_waveform` 输入，`vfc_decay_index` 只表示径向
`B_Z(R)` 剖面，不是时间衰减率。

TPC 的镜像参数与 VFC 的垂直场参数是位形层量；`vfc_current_fraction`（或
同义的新字段 `vertical_charge_separation_fraction`）则是单独的垂直电荷
分离电流闭合，默认严格为零，并且不由 `field_configuration="VFC"` 自动
打开。该系数即使在其他 provider 下也可显式设置，因此比较时应将“改位形”
和“开/关垂直电荷分离机制”分开。不要把 VFC 位形自动解释成一定存在
`I_vfc`，也不要把它与 EXL-50 文献的 `j_VTF` 缩写等同。

默认解析 provider 仍不是 Biot--Savart、自由边界 Grad--Shafranov 或真实场线追踪；可选的圆环细丝 Biot--Savart 后端见下节。生产计算应把线圈电流波形、二维 `B_R/B_Z/B_phi`、场线连接长度和壁命中率接入同一 provider 契约。

### `geometry.magnetic_backend` 与 `geometry.biot_savart_loops`（可选）

当需要从实际圆形线圈几何开始，而不是使用局部场系数时，可显式选择
`magnetic_backend: biot_savart`，并提供一个或多个理想圆形细丝线圈：

```yaml
geometry:
  field_configuration: VFC
  magnetic_backend: biot_savart
  biot_savart_loops:
    - name: PF_upper
      radius_m: 0.45
      z0_m: 0.20
      turns: 8
      current_waveform_A: [[0.0, 0.0], [0.0005, 120000.0]]
    - name: PF_lower
      radius_m: 0.45
      z0_m: -0.20
      turns: 8
      current_A: -60000.0
```

字段含义：`radius_m` 为线圈半径（m），`z0_m` 为线圈平面高度（m），
`current_A`/`current_waveform_A` 为每匝电流（A），`turns` 为正整数匝数，
`name` 必须唯一。波形区间外保持端点值；有限截面线圈应拆成多个具有实际
半径和高度的 loop。该 provider 使用椭圆积分计算轴对称场，仍不求解
自由边界 GS、3D 误差场或壁命中率，详见 [`docs/biot_savart.md`](biot_savart.md)。

`magnetic_backend=biot_savart` 时至少需要一个 loop，且不能同时填写
`coil_components`；反过来，填写 `biot_savart_loops` 也必须显式选择该
backend。这样可避免同一输入中混用两套线圈场定义。

### `geometry.coil_components`（可选）

这是一个保持向后兼容的解析叠加接口。列表非空时，
`make_magnetic_field()` 使用所选 `field_configuration` provider 作为基础，
再按每个分量的 `current_A`/`current_waveform_A` 线性叠加局部
`B_R/B_Z/B_phi`。分量字段为：

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `name` | string（必填且唯一） | 诊断和复现用名称 |
| `role` | `auxiliary`/`TPC`/`VFC` | 标签，不自动改变拓扑或电流机制 |
| `current_A` | float=0 | 未提供波形时使用的恒定电流；有波形时忽略 |
| `current_waveform_A` | `[[time_s,current_A],...]` | 分段线性电流波形，时间严格递增 |
| `B_R_T_per_A`、`B_Z_T_per_A`、`B_phi_T_per_A` | float=0 | 局部解析场系数，单位 T/A |

这是 `CompositeMagneticField` 的低阶 surrogate，不是线圈几何、Biot--Savart
或自由边界 GS 解；`role=TPC/VFC` 也不会自动等同于 `j_TPC/j_VTF`。真实
线圈场应通过 `MagneticFieldProvider` 注入（详见
[`docs/coil_superposition.md`](coil_superposition.md)）。

## `gas`

`species`（如 `D2`）、`pressure_Pa`、`Tn_K`、`ion_mass_kg`、`ionization_energy_eV`、`excitation_energy_eV`、`recombination_coeff_m3_s`、`attachment_coeff_m3_s`、`recycling_coefficient`、`pump_time_s`、`secondary_emission`，以及三个仅用于 surrogate 的速率缩放：`momentum_rate_scale_m3_s`、`ionization_rate_scale_m3_s`、`excitation_rate_scale_m3_s`。可选 `pressure_waveform_Pa=[[time_s,pressure_Pa],...]` 给出分段线性的中性气体目标压强；时间可以为负且必须严格递增，压强不得为负。它进入中性气体泵浦/补气平衡，并通过 `n_n=p/(k_B T_n)` 影响 `E/N`、碰撞、电离和激发。G-EQDSK 中的 `pressure` 是等离子体动力学压强，不能填入这里代替中性充气压强。接入截面数据库时应新增 `collision_provider`、`database_version` 和不确定度字段，并保持旧字段可追溯。

## `initial`

`ne_m3`、`ni_m3`、可选 `nn_m3`（省略则由 `p/(kT)` 计算）、`Te_eV`、`Ti_eV`、`Ip_A`、`n_hot_m3`、`hot_energy_J_m3`、`seed_rate_m3_s`。

## `rf`

`rf.enabled` 是所有 EC 系统的总开关；INI 模板写作
`[rf] switchOn = 'enabled'`。共享热尾参数是 `hot_loss_time_s`、
`hot_scattering_time_s`、`orbit_asymmetry` 和全局 `current_drive_sign`。

`rf.sources[]` 中每个源完全独立，字段包括 `name`、`enabled`、
`frequency_Hz`、`harmonic`、`power_W`、`absorption_fraction`、
`pulse{start_s,end_s,ramp_up_s,ramp_down_s}`、
`launch{R_m,Z_m,poloidal_angle_deg,toroidal_angle_deg,k_parallel_m_inv,mode,polarization}`、
`hot_fraction`、`hot_creation_energy_eV`、`current_drive_A_per_W` 和
`current_drive_sign`。INI 中用 `[rf_source:<name>]` 表示一个源，并将嵌套字段
展平为 `pulse_start_s`、`launch_R_m` 等。28 GHz O-mode 和 50 GHz X-mode
可以任意单独启用或同时启用；同时启用时总功率、热电子源和 RF 电流是各
启用源贡献之和，不要求两个源具有相同时序或功率。

`launch` 目前只进入可复现实验清单和未来射线/全波接口；模式/极化尚未进入
真实传播和吸收计算。`current_drive_A_per_W` 是显式经验替代项，默认值为零。
当 `sources` 为空时，顶层旧式单源字段仍用于兼容历史配置。

## `ray_tracing`

冷等离子体 Hamiltonian EC 射线的独立诊断配置。`enabled` 是未来把传播结果
反馈到 0D 功率方程的预留开关；显式执行 `raytrace-ec` 命令时无论其值为何
都会运行诊断。字段为 `dispersion_model='cold_stix'`、`step_m`、
`max_length_m`、`max_steps`、`spatial_difference_m`、
`wavevector_relative_step`、`resonance_tolerance`、`wkb_min_scale_ratio`、
`terminate_on_wkb_failure`，以及可选计算域 `R_bounds_m/Z_bounds_m`。
计算域不是第一壁，除非用户明确输入实测壁边界。完整方程、角度约定和终止
状态见 [`ec_ray_tracing.md`](ec_ray_tracing.md)。

## `ebw`

`switchOn='enabled'/'disabled'` 预留是否把该诊断接入工作流；显式执行
`analyze-oxb` 时始终运行。`density_profile='gaussian_cloud'` 使用物理 R-Z
密度云，而非磁通面：`center_R_m/center_Z_m`、`sigma_R_m/sigma_Z_m`、
`density_floor_m3` 和 `schematic_peak_density_m3` 定义其位置、宽度和密度。
`center_R_m=none` 时取 `geometry.R0_m`。`peak_to_average_factor` 只用于
`--average-ne` 到峰值的示意映射。

`cutoff_tolerance_fraction/uhr_tolerance_fraction` 是有限步长下识别 O cutoff
和 UHR 的相对容差；`minimum_ox_efficiency` 是冷 slab O-X 代理门槛；
`minimum_abs_n_parallel_for_cd` 只决定是否允许标记几何 EBWCD 候选。
`ebw_path_step_m/ebw_max_length_m` 控制示意路径，`R_bounds_m/Z_bounds_m`
只是计算域。上述参数不能替代真实 `n_e(R,Z,t)`、热色散或全波 X-B 模型。
完整边界见 [`ec_ray_tracing.md`](ec_ray_tracing.md)。

## `kinetic`

`0D-2V` 电子 Fokker--Planck诊断与可选电流反馈配置。显式 `solve-fp` 命令
始终运行。字符值须加单引号。

- `backend='reduced_0d2v'|'open_field_line_1d2v'`：体平均速度空间后端，或沿真实/外部追踪开场线的空间动理学后端；
- `coupling_mode='diagnostic_only'|'moment_feedback'|'time_dependent_feedback'`：
  独立诊断、分布矩响应表反馈或逐时刻全局/F-P双向固定点反馈；反馈模式要求
  `switchOn='enabled'`；
- `n_v_parallel/n_v_perp` 与 `v_parallel_max_m_s/v_perp_max_m_s`：均匀、
  cell-centred 柱速度网格；
- `time_step_s/steps`：独立 F-P 时间步和步数；
- `initial_drift_v_parallel_m_s`：初始漂移 Maxwellian；密度/温度来自
  `initial.ne_m3/Te_eV`；
- `collision_frequency_s` 与可选 `collision_temperature_eV`：OU 降阶碰撞；
  后者为 `none` 时使用 `initial.Te_eV`；
- `electric_field_V_m`：有符号平行电场；电子电荷取 `-e`；
- `loss_frequency_s`：当前为均匀 Krook 损失率；
- `rf_diffusion_model='zero'|'synthetic_gaussian'|'external_npz'|'genray_npz'`；
- `rf_diffusion_reference_power_W`：GENRAY NPZ未携带参考吸收功率时的显式补充；
- `rf_diffusion_model='genray_like_internal'`：不用外部文件，由 `[genray_like]`
  控制的内部冷射线/相对论共振核生成参考 `D_ql`；它不等同于完整 GENRAY。

`[genray_like]`：

- `cross_diffusion_fraction`：非对角 `D_parallel-perp` 相对于两对角项几何平均的
  系数，范围 `[-1,1]`，以保证二维扩散张量正半定。

`[kinetic]` 的高级选项：

- `coordinate_system='velocity'|'relativistic_momentum'`；后者用
  `u_parallel_max/u_perp_max` 设置 `p/(m_e c)` 网格。
- `momentum_conserving_field_particle='enabled'|'disabled'`：线性化 Coulomb
  场粒子修正，不等同于完整 Rosenbluth 算子。
- `orbit_model='no_orbit_average'|'trapped_passing_bounce_average'`：后者使用
  `orbit_B_min_T/orbit_B_max_T` 的 pre-CFS 镜像判据和驻留时间权重；不是闭合磁面
  上的定量 bounce average。

- `source_name`：`none` 表示按名义吸收功率组合所有已启用 RF 源；字符串则只选
  同名 `[rf_source:<name>]`；
- `resonance_width_fraction`：相对共振宽度 `Delta omega / omega`；
- `peak_absorption_coefficient_m_inv`：沿射线 `dP/ds=-2 alpha P` 的降阶峰值；
- `density_reference_m3` / `temperature_reference_eV` 及对应 exponent：吸收权重
  的显式缩放参考；
- `parallel_diffusion_floor`：对角降阶张量中平行扩散的最小比例；
- `minimum_kernel_peak`：速度网格未覆盖共振时的拒绝阈值；
- `adjoint_normalized_efficiency`：只用于独立电流量级诊断的无量纲 `zeta`，不
  修改 `D_ql` 或 F-P 电流；
- `cache_reference_diffusion`：是否缓存相同配置/速度网格的射线和参考 `D_ql`。
- 高斯模型用两个峰值、共振中心和宽度字段，单位分别为 `m2/s3` 和 `m/s`；
- 外部模型用 `rf_diffusion_file` 指向 NPZ，数组必须与配置网格同形。
- `feedback_reference_absorbed_power_W`、`feedback_reference_density_m3`、
  `feedback_density_ratio_cap` 和 `feedback_power_fractions` 控制响应表归一化；
- `feedback_replace_direct_rf/feedback_replace_hot_tail` 独立控制替换的经验通道。
- `time_dependent_iterations/time_dependent_relaxation/time_dependent_tolerance_A`
  控制逐时刻固定点迭代，`time_dependent_snapshot_count` 控制分布快照数。
- `spatial_fieldline_file`：`open_field_line_1d2v` 必填的 NPZ；必须含
  `s_m/R_m/Z_m/B_T/cross_section_area_m2`，代码不会合成一条“真实”场线。
- `spatial_dql_file`：功率、动量和网格元数据通过检查的空间非对角 `Dql` NPZ；
  有 RF 功率的 `time_dependent_feedback` 缺少此文件时直接报错。
- `spatial_sheath_model='none'|'quasineutral_ambipolar'`：可选空间双极场/Bohm鞘闭合。
- `spatial_ionization_from_global_rates` 和 `spatial_use_global_rf_power` 控制逐时刻输入；
  1D-2V 输出会替换全局模型的电离、吸收和电子壁损失项，避免重复计数。

高斯扩散仅用于趋势/符号测试，外部扩散只有经过波功率归一化后才能用于
定量电流。完整方程、CQL3D 对照和网格收敛要求见
[`fokker_planck.md`](fokker_planck.md)。

## `circuit`

`cs_mode='with_cs'|'without_cs'` 显式控制感应启动：前者使用
`loop_voltage_V/loop_voltage_waveform`，后者将有效环电压严格置零。
其余字段为 `loop_voltage_ramp_s`、`total_inductance_H`、可选
`plasma_length_m`、`conductivity_floor_S_m`、`resistance_cap_ohm`。
`effective_charge_override=none` 时使用逐电荷态壁模块给出的动态 `Z_eff`；
填写大于等于 1 的数值时覆盖它，适合受控扫描。

## `thresholds`

`breakdown_density_m3`、`preionization_density_m3`、`preionization_Te_eV`、`cfs_closed_fraction`、`persistence_steps`、`failure_density_m3`。这些是事件检测阈值，不是自然常数；应通过诊断不确定度分析标定。

## `numerics` 与 `outputs`

`numerics`：`t_start_s`、`t_end_s`、`dt_s`、`solver`（`BDF/RADAU/RK45`）、`rtol`、`atol`、可选 `max_step_s`。`t_start_s` 可以为负，但必须满足 `t_end_s>t_start_s`；负时刻适合表示以 `Ip` 出现或其他实验触发为零点之前的注气和 EC 投入。

`startup_window` 只定义时间的物理语义，不重复控制积分器：
`definition='explicit_time'|'event_relative'`，`reference_event` 可为
`'absolute_clock'|'ec_on'|'gas_on'|'first_Ip'|'breakdown'|'custom'`，
`end_condition` 可为 `'integration_end'|'first_persistent_cfs'|'diagnostic_marker'|'custom'`。
自定义事件必须填写对应说明；CFS 终止还可给 `cfs_persistence_s`。因此 EXL-50U
示例中的 0--100 ms 不会被解释成其他装置的固定启动时间。

`outputs`：`directory`、可选 `cadence_s`、`write_netcdf`、`write_csv`。
`write_current_profile='enabled'` 输出 `rho_geo` 剖面 PNG/NPZ；
`write_pdf_report='enabled'` 将所有主结果图写入 `pdf_report_filename`。
`profile_radial_points/profile_snapshot_count` 控制剖面分辨率和快照数。

## `current_profile`

0D 模型没有真实径向输运，因此这里只做严格归一化的剖面重建。
`coordinate='geometric_r_over_a'` 表示 `rho_geo=r/a`；CFS 前不得解释为磁通坐标。
每个机制的 `[p,q]` 形状代表 `rho^(2p)*(1-rho^2)^q`，并按极向截面积分到
对应的因果 L-R 电流贡献。字段为 `ohmic_shape/rf_shape/tail_shape/vertical_shape`
和数值闭合残差使用的 `residual_shape`。

## `threshold_scan`

配置压强、有效电荷、EC 功率和 CS 模式的笛卡尔积扫描。每个工况都与同条件
的零 EC 功率基线比较，避免把 CS 欧姆电流误记为 EC 驱动。字段和成功判据见
[`threshold_scan.md`](threshold_scan.md)。
`refine_enabled/refine_power_tolerance_W/refine_max_iterations` 控制失败--成功功率
区间的二分细化。
`kinetic_coupling_mode='inherit'|'moment_feedback'|'time_dependent_feedback'`
显式控制大扫描采用主案例耦合模式还是较快响应表；报告必须显示该选择。

## `experimental_startup_prior`

这是实验锚点和模型结果的比较层，不会修改ODE。`switchOn`控制是否在正式PDF
输出该页。EXL-50U模板包括弱CS电流、弱CS是否按无CS分类、稳健启动功率上界、
70 kW气压上限锚点、80/220 kW最佳Ip气压、300 kW参考、
`importance_order=['power','pressure','impurity']`，以及H-alpha峰是否被视为
终止必要条件。没有CS互感和 `dI_CS/dt` 时，弱CS线圈电流不会自动转换为环电压。

## `validation`

设备级定量质量门。`intended_use='diagnostic_only'` 是默认值；改为
`'quantitative_device_prediction'` 并不会自动放行。必须逐项声明实际线圈、
气压、杂质、RF 波束/Dql、壁几何和轨道损失输入，并给出成功/失败炮数量、
启动灵敏度/特异度以及 Ip 轨迹相对误差。详见
[`device_prediction_quality.md`](device_prediction_quality.md)。

## `aspect_ratio_scan`

`R0_values_m` 与 `minor_radius_values_m` 是两个独立轴；`combination='cartesian'`
取笛卡尔积，`'paired'` 按位置配对。`scale_rf_launch_with_geometry` 控制是否保持
归一化发射位置。每点必须满足 `R0_m > minor_radius_m`。该扫描用于研究低纵横比、
比压代理和多共振层的协同，不会覆盖设备级质量门。

完整可运行示例见 [`configs/minimal.yaml`](../configs/minimal.yaml)。

## `species_wall`（可选 DYON-like 模块）

默认 `enabled=false`，因此旧案例不增加状态或功率项。启用时：

- `atomic_backend='analytic_surrogate'|'openadas_adf11'`：前者仅用于软件联调；
  后者读取带来源 URL、SHA-256 和单位声明的正式 ADF11 manifest；
- `atomic_manifest`：`openadas_adf11` 必填，当前仓库提供带来源 URL 和 SHA-256 的
  `data/openadas/adf11_h_96/manifest.json`；受 OPEN-ADAS 分发条款限制，原始 `.dat`
  文件不随仓库发布，需按同目录 `README.md` 获取并校验；
- `atomic_bounds='raise'|'clip'`：表格范围外默认拒绝，`clip` 必须在报告中披露；
- `species[]`：`symbol`、`role=fuel|impurity`、`atomic_number`、`max_charge`；
- impurity 必须给出长度为 `max_charge+1` 的 `initial_charge_densities_m3`；fuel 的初值来自 `initial.nn_m3/ni_m3`，不得重复给出；
- 可选 `ionization_energies_eV` 长度等于 `max_charge`；
- `ionization_rate_scale_m3_s`、`recombination_rate_scale_m3_s`、`radiation_coefficient_W_m3` 是解析后端标定量；最后一个字段的量纲是 W m³；
- `wall_loss_multiplier` 缩放杂质离子壁损失，`neutral_pump_time_s` 控制杂质中性抽气；
- `wall_materials[]` 使用 `name` 和 `species` 引用元素，并配置 `initial_inventory_m2`、`recycling_coefficient`、可选 `desorption_time_s`；
- 溅射参数为 `physical_sputtering_yield`、`physical_sputtering_threshold_eV`、`chemical_sputtering_yield` 和 `incident_energy_factor`，产额默认均为零。

OpenADAS H-96 后端只允许 H/D/T 的 `q=0/1` 原子电离、复合和辐射；它不包含
H2/D2 解离、分子离子或振转态。首版最多一个 fuel，并且 fuel 必须是
`max_charge=1`；每个元素最多一个 wall material。完整示例见
[`configs/dyon_species_wall.yaml`](../configs/dyon_species_wall.yaml)，方程和 ADAS
provider 接口见 [`dyon_species_wall.md`](dyon_species_wall.md)。

## `charge_closure`

- `mode=legacy|quasineutral|ambipolar`：旧损失、严格净边界电荷约束或玻尔兹曼双极势闭合；
- `initialize_ne_from_ions=true`：非旧模式下由 `sum(q*n_q)` 设置初始 `ne` 并保持输入 `Te`；
- `potential_min_over_Te`、`potential_max_over_Te`：双极势 `e phi_a/Te` 的下/上限；
- `flux_floor_m3_s`：反求对数时的正通量下限。

完整符号约定和失效条件见 [`charge_closure.md`](charge_closure.md)。
