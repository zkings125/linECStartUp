# 输出参数说明

## `state.nc`

当前 v0.1 是一维 `time` 数据集；后续空间模型应扩展为 `time,s` 或 `time,R,Z`，不改变变量名和单位属性。

由于 Windows 中文路径下部分 `netCDF4` 版本存在编码兼容问题，本项目写出端使用 NetCDF3/scipy 后端；读取时可显式使用 `xarray.open_dataset("state.nc", engine="scipy")`。复制到 ASCII 路径后也可用默认后端读取。

| 变量 | 单位 | 含义 |
|---|---|---|
| `ne_m3`, `ni_m3`, `nn_m3` | m⁻³ | 电子、离子、中性密度 |
| `Te_eV`, `Ti_eV` | eV | 体平均电子/离子温度 |
| `Ip_A` | A | 积分电流状态 |
| `n_hot_m3` | m⁻³ | 热尾矩替代密度 |
| `hot_mean_energy_eV` | eV | 热尾平均能量 |
| `E_loop_V_m` | V/m | 感应环向场替代 |
| `EoverN_Td` | Td | 约化场 |
| `K_ion_m3_s`, `K_exc_m3_s` | m³/s | 反应率闭合 |
| `nu_ion_s`, `nu_loss_s`, `net_growth_s` | s⁻¹ | 电离、损失和净增长率 |
| `townsend_alpha_eff_m` | m⁻¹ | 低阶 Townsend 指标 |
| `Dreicer_ratio` | 1 | `E/E_D` 预警 |
| `omega_ce_over_nu` | 1 | 电子回旋频率/总碰撞频率 |
| `lambda_D_over_a` | 1 | Debye 长度/小半径 |
| `mean_free_path_over_a` | 1 | 电子平均自由程/小半径 |
| `ionization_fraction` | 1 | `n_e/(n_e+n_n)` |
| `neutral_pressure_Pa`, `target_pressure_Pa`, `pL_Pa_m` | Pa, Pa, Pa m | 由当前中性密度反算的压强、输入压强波形目标值及 Paschen 初筛量 |
| `P_ohm_W_m3`, `P_rf_W_m3`, `P_rf_bulk_W_m3`, `P_ion_W_m3`, `P_exc_W_m3`, `P_rad_W_m3`, `P_wall_W_m3`, `P_loss_W_m3`, `P_balance_W_m3` | W/m³ | 功率收支；`P_rf_W_m3` 为总吸收，`P_rf_bulk_W_m3` 为扣除热尾分配后的体电子功率 |
| `sigma_S_m`, `R_plasma_ohm` | S/m, Ω | 电导与等效电阻 |
| `I_ohmic_A`, `I_bulk_A`, `I_rf_A`, `I_kinetic_A`, `I_orbit_A`, `I_tail_A`, `I_vertical_charge_A`, `I_tpc_A`, `I_vfc_A`, `I_vtf_A`, `I_drive_A`, `I_component_sum_A` | A | 电流机制分解；反馈模式下 `I_rf_A=I_kinetic_A`。规范名是 `I_tail`（热尾优先约束/损失不对称）和 `I_vertical_charge`（垂直电荷分离）。别名不是 TPC/VFC 位形的独立方程 |
| `current_drive_model` | string | `empirical_rf_hot_tail` 或 `fp_moment_response_table`，用于审计电流闭合来源 |
| `rf_power_absorbed_W`, `rf_accessibility` | W, 1 | RF 源诊断 |
| `rf_source_count` | 1 | 已配置的独立 RF 源数量，包括当前关闭的源 |
| `rf_<i>_name`, `rf_<i>_frequency_Hz` | string, Hz | 第 `i` 个 RF 源的名称和频率；顺序与输入一致 |
| `rf_<i>_power_absorbed_W`, `rf_<i>_power_density_W_m3`, `rf_<i>_hot_power_density_W_m3` | W, W/m³, W/m³ | 第 `i` 个源的吸收功率、体积功率和热尾功率；关闭或不在脉冲窗口时为零 |
| `rf_<i>_pulse_factor`, `rf_<i>_accessibility`, `rf_<i>_current_drive_A` | 1, 1, A | 第 `i` 个源的时序因子、低密度可达性替代值和 RF 电流贡献 |
| `rf_<i>_launch_R_m`, `rf_<i>_launch_Z_m`, `rf_<i>_mode`, `rf_<i>_polarization` | m, m, string, string | 第 `i` 个源的发射点、模式和极化记录 |
| `rf_<i>_omega_ce_over_omega_rf`, `rf_<i>_critical_density_m3` | 1, m⁻³ | 第 `i` 个源的参考回旋频率比和冷等离子体临界密度 |
| `omega_ce_over_omega_rf` | 1 | 参考环向场与 RF 频率的回旋频率比 |
| `B_vertical_T` | T | 当前外部垂直场（静态值或波形插值） |
| `B_R_T`, `B_Z_T`, `B_phi_T`, `B_magnitude_T` | T | RF 发射点（未指定时为磁轴）处的 provider 场分量和模长 |
| `field_configuration` | string | 基础 provider 名；有线圈叠加时为 `composite[<base>]` |
| `base_field_configuration` | string | 复合 provider 使用的基础位形（仅复合场输出） |
| `field_provider` | string | provider 实现标签（例如 `composite_analytic_coils`） |
| `coil_component_count` | 1 | 叠加线圈分量数 |
| `coil_current_total_A` | A | 所有解析分量的代数电流和 |
| `coil_B_R_axis_T`、`coil_B_Z_axis_T`、`coil_B_phi_axis_T` | T | 叠加分量在磁轴的场和 |
| `Bp_edge_base_T`、`coil_Bp_edge_T` | T | 基础 provider 的边界极向场，以及线圈分量边界场的诊断量；后者不直接计入当前 CFS surrogate 的等离子体 `Bp` |
| `coil_<i>_name`、`coil_<i>_role`、`coil_<i>_current_A` | string/A | 第 `i` 个分量的名称、角色和当前电流 |
| `loop_count`、`loop_current_total_A` | 1/A | Biot--Savart 圆环数及代数安匝电流和 |
| `loop_B_R_axis_T`、`loop_B_Z_axis_T`、`loop_Bp_edge_T` | T | Biot--Savart 圆环叠加在磁轴/边缘的场诊断 |
| `loop_edge_singularity_count` | 1 | 边缘代表采样点落在理想细丝上的线圈数；非零时 `loop_Bp_edge_T` 为 NaN，需改用有限截面或避开采样点 |
| `mirror_ratio`, `trapped_fraction`, `decay_index` | 1 | 位形 provider 的镜像比、捕获粒子比例和垂直场衰减指标 |
| `connection_length_factor`, `field_loss_multiplier`, `mirror_confinement_factor`, `tail_confinement_metric` | 1 | 位形对开场线连接长度/热尾约束的显式 surrogate 指标 |
| `closed_fraction`, `candidate_radius_m`, `Bp_edge_T`, `connection_length_effective_m` | 1, m, T, m | CFS 替代诊断 |
| `topology_confidence` | 1 | 当前拓扑方法的可信度标签（surrogate 低） |
| `stage` | string | 状态机阶段 |

`j_TPC`/`j_VTF` 是 EXL-50 论文中的电流密度/限制器诊断记号；本文件的
`I_tail_A`/`I_vertical_charge_A` 是体积分电流（A），二者必须通过壁/限制器
几何和诊断前向模型映射，不能按字段名直接替换。

另外，`diagnostics.csv/state.nc` 还会保留以下完整的反应源项和数值量，避免只保存汇总指标：

| 变量 | 单位 | 含义 |
|---|---|---|
| `electron_energy_J_m3`, `hot_energy_J_m3` | J/m³ | 被积分的体积能量状态 |
| `nu_en_s`, `nu_ei_s` | s⁻¹ | 电子-中性/电子-离子碰撞频率 |
| `K_ion_hot_m3_s` | m³/s | 热尾等效电离系数 |
| `S_ion_m3_s`, `S_ion_bulk_m3_s`, `S_ion_hot_m3_s` | m⁻³/s | 总、冷电子、热尾电离源 |
| `S_rec_m3_s`, `S_att_m3_s` | m⁻³/s | 复合和附着损失 |
| `S_e_wall_m3_s`, `S_i_wall_m3_s`, `S_secondary_m3_s` | m⁻³/s | 电子/离子壁损失及二次电子源 |
| `S_seed_m3_s` | m⁻³/s | 外部种子对（简化源） |
| `dIp_dt_A_s` | A/s | 电流方程右端项 |
| `omega_ce_over_omega_rf` | 1 | 回旋频率/发射频率参考比 |

属性中会写入 `schema_version`、`model`、方程文档路径和拓扑替代说明。

## `events.json`

每个事件包含：

```json
{
  "type": "breakdown | preionization | cfs_candidate | cfs | startup_failed",
  "time_s": 0.0001,
  "stage_from": "vacuum",
  "stage_to": "breakdown",
  "metrics": {"ne_m3": 1.0e16, "net_growth_s": 123.0}
}
```

事件时间是输出采样点精度；接入生产模型时应使用连续事件定位并给出时间不确定度。

## 其他文件

- `diagnostics.csv`：与 `state.nc` 相同的时间序列，带 `time_s` 列，便于 Excel/Pandas 快速检查；
- `summary.json`：求解器状态、最终阶段、最终诊断和配置哈希；
- `config.resolved.json`：Pydantic 校验后的完整配置（含推导体积/壁面积）；
- `environment.json`：Python/平台、代码版本、配置哈希和随机种子；
- `run.log`：最小可审计日志。高保真版本应在此基础上增加每步收敛和守恒残差。
- `current_profile_evolution.png/.npz`：启用剖面输出时生成；`rho_geo=r/a`，
  NPZ 中各机制电流密度对极向截面积分回对应 0D 因果电流；
- `startup_report.pdf`：启用 PDF 时生成的统一多页报告，包含 `Ip`、阶段、机制、
  径向重建、注气/`Z_eff` 和功率平衡；
- `threshold_scan.json`：保存粗扫/细化功率点及最后失败--首次成功功率区间。

## 启用 `species_wall` 后的动态变量

变量名按元素符号和电荷态生成：

| 模式/变量 | 单位 | 含义 |
|---|---|---|
| `n_<S><q>_m3` | m⁻³ | 元素 `S` 的第 `q` 电荷态密度；fuel 的 q=0/1 映射自 `nn/ni` |
| `wall_inventory_<S>_m2` | m⁻² | 该元素的壁面滞留库存 |
| `S_ion_<S><q>_m3_s`, `S_rec_<S><q>_m3_s` | m⁻³ s⁻¹ | 相邻电荷态的电离与复合源项 |
| `S_wall_<S>_m3_s`, `S_pump_<S>_m3_s` | m⁻³ s⁻¹ | 壁损失与中性抽气 |
| `S_recycle_<S>_m3_s`, `S_sputter_<S>_m3_s` | m⁻³ s⁻¹ | 再循环/解吸返回源与溅射源 |
| `P_rad_<S><q>_W_m3`, `P_rad_<S>_W_m3` | W m⁻³ | 分电荷态及元素总辐射 |
| `P_rad_species_W_m3`, `P_ion_species_W_m3` | W m⁻³ | 多物种总辐射和杂质电离功率 |
| `P_rad_base_W_m3` | W m⁻³ | 原 7 状态连续谱 surrogate，便于区分新增项 |
| `electron_density_quasineutral_m3` | m⁻³ | `sum(q*n_s,q)` |
| `charge_balance_residual_m3` | m⁻³ | `ne-sum(q*n_s,q)`，只诊断、不强制归零 |
| `Z_eff` | 1 | `sum(q²*n_s,q)/sum(q*n_s,q)` |
| `species_particle_balance_residual_m3_s` | m⁻³ s⁻¹ | 扣除显式抽气和固体溅射源后的模块核子守恒残差 |
| `species_particle_balance_relative` | 1 | 上述残差除以壁/抽气/溅射通量尺度，判断浮点相消时优先使用 |
| `atomic_rate_backend` | string | 原子率 provider 标签 |

## 电荷闭合诊断

| 变量 | 单位 | 含义 |
|---|---|---|
| `charge_closure_mode` | string | `legacy`、`quasineutral` 或 `ambipolar` |
| `S_e_wall_free_m3_s`, `S_e_wall_target_m3_s`, `S_e_wall_m3_s` | m⁻³ s⁻¹ | 自由、连续性目标和实际电子边界损失 |
| `S_positive_charge_wall_m3_s` | m⁻³ s⁻¹ | 主离子加杂质 `q` 加权正电荷壁损失 |
| `ambipolar_potential_V` | V | 体平均等效双极势 |
| `ambipolar_potential_over_Te` | 1 | `e phi_a/Te`；Te 以 eV 表示时数值为 `phi_a/Te` |
| `charge_continuity_residual_m3_s` | m⁻³ s⁻¹ | `d(ne-sum(qn))/dt` 的边界闭合残差 |
| `charge_continuity_relative` | 1 | 连续性残差除以相关壁通量尺度 |
| `charge_balance_relative` | 1 | 状态层面的相对电荷不平衡 |
| `ambipolar_flux_limited` | 1 | 势垒达到上下限时为 1 |

NetCDF 全局属性 `species_wall_note` 会声明模块是否启用，并明确解析后端不是 ADAS。

## `raytrace-ec` JSON

该独立命令不修改 `state.nc`，而是输出单个 RF 源在指定 `t_s/Ip/ne` 快照上的
传播诊断。摘要字段包括 `status/reason`、`source_name`、`frequency_Hz`、
`harmonic`、`mode`、`reached_resonance`、`path_length_m` 和
`launch_direction_source`。路径点包括 `R_m/Z_m/phi_rad`、
`k_R_m_inv/k_Z_m_inv/k_phi_m_inv`、`B_T`、`electron_density_m3`、
`refractive_index`、`n_parallel/n_perpendicular`、`resonance_mismatch`、
`hamiltonian` 与 `wkb_scale_ratio`。使用 `--no-path` 只保留最终点和摘要。

`equilibrium_model` 明确写为 `none; MagneticFieldProvider only`；这表示结果没有
调用 GS，不表示磁场是已验证的真实启动场。终止状态和适用范围见
[`ec_ray_tracing.md`](ec_ray_tracing.md)。

## `analyze-oxb` JSON

该命令同样不修改 0D 状态。`O_cutoff/UHR/resonance` 给出候选点的
`R_m/Z_m`、路径长度、局域密度、磁场和残差；`O_cutoff_density_m3` 是频率
确定的冷 O 模截止密度。`ox_efficiency_estimate` 是降阶冷 slab 角度失配
代理，`xb_efficiency_estimate` 和 `combined_efficiency_estimate` 当前固定为
`null`，防止把几何层识别误报成全波效率。

`ebw_resonance_geometric_candidate` 表示示意路径到达选定谐波层附近；只有再
满足 `abs(n_parallel_at_O_cutoff)` 门槛时，`ebwcd_candidate` 才为真，并且
`ebwcd_assessment` 仍明确写成 `geometric_candidate_only`。
`current_drive_direction_sign` 仅记录 `N_parallel` 的符号（`+1/-1`），不是
电流大小或最终环向电流方向预测。`current_drive_model` 会明确声明：仅对该
几何 `analyze-oxb` 命令而言，热色散和准线性/Fokker--Planck 闭合尚未接入。
外部经验证全波场可通过独立的 `project-fullwave-dql` 命令生成空间 `Dql`，
但不会反向把几何候选结果改写成定量转换效率。

`density_source/density_profile` 记录示意密度来源，`incident_O_ray` 保留入射
O 射线。`equilibrium_model='none; MagneticFieldProvider only'` 明确该诊断在
pre-CFS 阶段没有使用 GS/EFIT 磁面。

## `solve-fp` JSON 与 NPZ

该命令独立于全局 `state.nc`。JSON 保存 `initial_moments/final_moments`，其中
包括密度、平均平行速度、能量密度、等效温度、`J_parallel`、截面积分
`current_parallel_A` 以及平行/垂直压力。还保存速度网格、实际推进时间、
`max_abs_particle_balance_relative`、`max_linear_residual_relative`、
`minimum_f` 和 `rf_diffusion_source`。

`current_parallel_A` 尚未做 `B_phi/B` 空间投影，不应直接当作实验环向 `Ip`。
`rf_diffusion_source='synthetic_gaussian_resonance'` 时，JSON 的警告字段明确
禁止把结果解释为功率归一化 EC/EBWCD。

可选 NPZ 包含最终 `f_e[n_v_parallel,n_v_perp]`、初始 `f_initial`、`v_parallel_m_s`、
`v_perp_m_s` 和标量 `time_s`。外部扩散 NPZ 使用不同契约：必需
`D_parallel_m2_s3/D_perp_m2_s3`，可选携带两个速度坐标供严格核对。

启用 `time_dependent_feedback` 时，每个运行目录新增
`kinetic_time_history.npz`，包含逐时刻有功率/零Dql/增量电流矩、动理学温度、
平行/垂直压强矩、速度网格、电子分布快照、扩散来源和固定点残差。正式EXL
报告另写 `stage_mechanisms.json`；每个案例固定输出四个诊断窗口，即使物理
判据未达到也会以 `available=false`/`not reached` 明示，不会伪造阶段转换。

## 设备质量门与环径比扫描 JSON

`assess-device-readiness` 输出 `prediction_level`、输入/验证完整性、
`startup_decision_authorized`、`current_prediction_authorized`、缺失输入、失败指标
以及允许/禁止的结果类型。EXL-50U `manifest.json` 内嵌同一对象。

`aspect-ratio-scan` 输出每个 `(R0_m, minor_radius_m)` 点的 `aspect_ratio`、
`inverse_aspect_ratio`、`peak_thermal_beta`、可达共振层、启动共振诊断和全局启动
摘要。顶层 `cross_device_quantitative_authorized` 当前固定为 `false`，因为一个
基准配置的质量门不能替代每个真实装置各自的炮级验证。
