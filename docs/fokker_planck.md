# 0D-2V 电子 Fokker--Planck 后端

## 目的与边界

`ecstartup.physics.fokker_planck` 在无磁面坐标下推进
`f_e(v_parallel,v_perp,t)`，电流不是由经验 A/W 直接指定，而由分布一阶矩

\[
J_\parallel=q_e\int v_\parallel f_e\,d^3v,
\qquad d^3v=2\pi v_\perp dv_\perp dv_\parallel
\]

计算。`current_parallel_A=J_parallel*A_cross_section` 使用
`A_cross_section=V/(2*pi*R0)`；pre-CFS 阶段若要得到真正环向等离子体电流，
还必须加入局域磁场方向投影与空间积分。因此该值是平行电流截面积分诊断，
不能直接替代实验 `Ip`。

后端可选择非相对论速度网格或相对论动量网格
`u=p/(m_e c)`。后者用精确 `gamma=sqrt(1+u_parallel^2+u_perp^2)`、
`v=cu/gamma` 和速度空间 Jacobian `d3v=d3u*c^3/gamma^5` 计算矩。
扩散张量支持 `D_parallel`、`D_perp` 和对称非对角 `D_cross`，并检查
`D_parallel*D_perp-D_cross^2>=0`。它仍是 0D-2V 降阶模型，不求解 GS，
也不是完整 CQL3D。

## 方程与数值格式

推进方程为

\[
\frac{\partial f}{\partial t}
=-\nabla_v\cdot(\mathbf a f-\mathbf D\nabla_v f)
-\nu_{loss}f+S,
\]

其中 `a_parallel=q_e E_parallel/m_e-nu_c v_parallel`、
`a_perp=-nu_c v_perp`，碰撞扩散为
`D_c=nu_c*T_c/m_e`。RF 项由外部
`D_parallel_RF`、`D_perp_RF` 和 `D_cross_RF` 加入二维扩散张量。
`momentum_conserving_field_particle='enabled'` 增加线性化场粒子修正，抵消
test-particle OU 拖曳对整体电子漂移动量的一阶损失。该闭合不是完整的
Rosenbluth 势或非线性 `C_ee+C_ei` 算子。

独立的 `NonlinearLandauRosenbluth2V` 已实现非相对论、轴对称 2V 的非线性
Landau 积分，并提供 `H/G` Rosenbluth 势诊断以及粒子、平行动量、能量离散
不变量投影。它可在 `OpenFieldLineKinetic1D2V.advance()` 中逐空间单元算子分裂
调用。该参考实现复杂度为 `O(Nv^2 Nphi)`，不等同于相对论
Braams-Karney 算子，也不能与旧标量碰撞频率同时开启。相对论网格可另选
`RelativisticBraamsKarney2V`：它以等价的 Beliaev-Budker 协变积分计算非线性
`Cee`，并加入无限重离子 Lorentz `Cei`，同时投影相对论不变量。

`orbit_model='trapped_passing_bounce_average'` 使用显式输入的 `B_min/B_max`
和 `lambda B_max>=1` 判据划分俘获/通过粒子，并按驻留时间对 `Dql` 加权。
pre-CFS 没有真实闭合磁面，因此这只是 open-field/mirror surrogate；形成 CFS
后才应切换到沿真实轨道的 bounce average。

速度空间采用 cell-centred 柱坐标有限体积和零通量边界。隐式步为
`(I-dt*L)f^(n+1)=f^n+dt*S`。漂移-扩散面通量使用 Chang--Cooper 指数拟合，
使解析 Ornstein--Uhlenbeck 碰撞对所对应 Maxwellian 保持离散稳态，同时
保留纯平流的迎风极限。每步输出粒子平衡、最小分布值和线性残差。

## 与 CQL3D 的对应关系

本实现参考 CQL3D 公共源码的模块职责，而未复制其 GPLv3 Fortran：

| 本项目职责 | CQL3D 参考模块 | 当前差异 |
|---|---|---|
| 碰撞漂移/扩散 | `cfpcoefr.f`, `cfpcoefn.f` | 默认 OU；另有直接积分相对论 `Cee+Cei` 参考后端，但尚非生产级快速五势求解器 |
| RF 准线性扩散 | `coefrfad.f` | 接收对角二维数组；不自行从波功率生成系数 |
| 平行电场 | `coefefad.f` | 均匀 `E_parallel` |
| 系数装配 | `coefstup.f` | 0D-2V 速度网格，无磁面平均 |
| 隐式推进/边界 | `impavnc.f` | 稀疏隐式 Chang--Cooper、零通量速度边界 |
| 损失 | `losscone.f` | 标量/二维 Krook 频率，不含真实轨道损失锥 |
| 密度/能量/电流矩 | `diaggnde.f`, `ampfar.f` | 柱速度空间矩，无空间壳层积分 |
| 守恒诊断 | `diag.f`, `tdoutput.f` | 每步粒子平衡和线性残差 |

CQL3D 主引用见 `docs/references.bib` 的 `Harvey1993CQL3D`。公开源码核对自
[`compxco/cql3d`](https://github.com/compxco/cql3d) commit
`3565fee6974deff69371321d8a053593eae1562d`；其 Fortran 源码为 GPLv3，当前
Python 求解器只参考公开方程与模块职责，没有复制实现。

## 运行

编辑 `configs/linec_parameters.ini` 的 `[kinetic]` 后执行：

```powershell
python -m ecstartup solve-fp configs/linec_parameters.ini `
  --output runs/fp_demo/summary.json `
  --distribution-output runs/fp_demo/distribution.npz
```

可复现的非定量高斯扩散演示及三联图：

```powershell
python -m ecstartup solve-fp configs/fp_synthetic_demo.ini `
  --output runs/fp_synthetic/summary.json `
  --distribution-output runs/fp_synthetic/distribution.npz
python tools/plot_fp_distribution.py runs/fp_synthetic/summary.json `
  runs/fp_synthetic/distribution.npz runs/fp_synthetic/fp_distribution_evolution.png
```

显式命令即使 `switchOn='disabled'` 也运行。NPZ 输出包含 `f_e`、
`v_parallel_m_s`、`v_perp_m_s` 和
`time_s`。

## 与全局电流方程耦合

设置 `switchOn='enabled'` 和 `coupling_mode='moment_feedback'` 后，代码在
全局积分开始前按 `feedback_power_fractions` 分别求解 F-P。每个功率点都减去
`D_ql=0` 的分布矩电流，因此初始漂移和平行感应场不会被重复记为 EC 电流。
吸收功率以 `feedback_reference_absorbed_power_W` 归一化，ODE 右端只插值响应表，
并按 `n_e/feedback_reference_density_m3` 缩放；缩放受
`feedback_density_ratio_cap` 限制。

`feedback_replace_direct_rf='enabled'` 用矩电流替换经验 `A/W` 项；
`feedback_replace_hot_tail` 独立决定是否关闭开场线优先损失的热尾代理。当前
0D-2V求解器没有空间/轨道损失坐标，因此默认保留该通道并单独输出，不能把
两者相加后称为完整CQL3D预测。

设置 `coupling_mode='time_dependent_feedback'` 后，代码使用外层固定点迭代：

1. 全局ODE给出每个输出时刻的 `ne/Te/E_parallel/nu_en/nu_ei/nu_loss/P_abs`；
2. 有功率和零 `D_ql` 两套分布沿同一时间轴因果推进；
3. 两者电流矩之差作为EC增量回灌下一次全局L-R求解；
4. 迭代到 `time_dependent_tolerance_A` 或达到 `time_dependent_iterations`。

`time_dependent_relaxation` 控制外迭代松弛，`time_dependent_snapshot_count`
控制保存到 `kinetic_time_history.npz` 和PDF的分布快照数。F-P状态不会在BDF
右端函数内更新，因为隐式求解器会重复并乱序调用右端函数。

`rf_diffusion_model` 有五种值：

- `'zero'`：`D_ql=0`，只检查电场、碰撞和损失；
- `'synthetic_gaussian'`：使用局域高斯扩散做符号和敏感性测试；
- `'external_npz'`：读取 `D_parallel_m2_s3`、`D_perp_m2_s3` 和可选
  `D_cross_m2_s3`，可选同时读取
  `v_parallel_m_s`、`v_perp_m_s` 并严格核对网格。
- `'genray_npz'`：在上述数组和速度网格之外，要求标量
  `reference_absorbed_power_W`（也接受 `absorbed_power_W`），或在配置中给出
  `rf_diffusion_reference_power_W`。逐时刻使用
  `D_ql(t)=D_ql,ref*P_abs(t)/P_ref`。若文件值与配置值不一致，程序拒绝运行。
- `'genray_like_internal'`：调用 linEC 自己的冷 Stix Hamilton 射线、相对论
  回旋谐波共振核和沿程降阶吸收，再以初始 Maxwellian 功率矩归一化 `D_ql`。
  该后端不依赖 GS/EFIT，但不是完整 GENRAY；细节和官方回归见
  `docs/genray_integration.md`。

这里“不依赖 GS/EFIT”不等于“无需空间背景”。内部生成器仍需要逐时刻
`B(R,Z,t)`、密度/温度模型和发射参数；区别只是这些量直接在实验室坐标中
给出，而不是从闭合磁面的磁通坐标获得。用以下命令检查证据等级：

```powershell
python -m ecstartup assess-dql configs/linec_parameters.ini
python -m ecstartup assess-dql configs/linec_parameters.ini `
  --dql-file runs/fullwave/spatial_dql.npz
```

输出中的 `pre_cfs_compatible=true` 只说明坐标和输入允许在 CFS 前使用。
`quantitative_eligible=true` 还要求功率归一化、明确的电磁 EC/EBW 分支、全波
网格与功率门；对 EBW 还要求 `mode_decomposition_verified=true`。只有
`ebw_out_W` 而没有本征模或波数谱分解时，代码标记为
`unresolved_fullwave`，不会称为 EBWCD。

GENRAY NPZ最小契约为：

```text
D_parallel_m2_s3[n_v_parallel,n_v_perp]
D_perp_m2_s3[n_v_parallel,n_v_perp]
D_cross_m2_s3[n_v_parallel,n_v_perp]  # 可选；缺失时为零
reference_absorbed_power_W scalar
v_parallel_m_s[n_v_parallel]  # 推荐，存在时严格核对
v_perp_m_s[n_v_perp]          # 推荐，存在时严格核对
```

这是GENRAY `D_ql`导出适配层和线性功率归一化，不是原生 NetCDF。原生
`genray.nc` 由 `inspect-genray` 读取；标准文件不含这里要求的二维速度空间
扩散数组。没有真实GENRAY文件时不得把
`synthetic_gaussian`结果称为GENRAY结果。

高斯代理的参考功率只是用户指定的灵敏度归一化，不能报告可信的 EC/EBWCD
安培数。生产链应由
GENRAY/CQL3D 接口、热色散射线或全波 O-X-B 程序给出带功率守恒的 `D_ql`。

对于外部 FEM/FDTD 全波输出，先检查端口功率预算，再投影到开放场线网格：

```powershell
python -m ecstartup inspect-fullwave fullwave_result.npz
python -m ecstartup project-fullwave-dql fullwave_result.npz open_fieldline.npz `
  configs/linec_parameters.ini --output runs/fullwave/spatial_dql.npz `
  --summary runs/fullwave/summary.json
```

空间输出形状为 `[n_s,n_parallel,n_perp]`，包含非对角项、场线路径、速度或
相对论动量网格、求解器身份、功率预算和 O-X-B 转换效率。它不读取 GS/EFIT。
全波 NPZ 与场线 NPZ 的完整字段见 [`high_fidelity_backends.md`](high_fidelity_backends.md)。

## 网格选择

默认 `81 x 64`、速度上界 `4e6 m/s` 针对 `Te约0.2 eV` 的冷启动分布；默认
离散矩温度误差约 0.3%。研究数十至数百 eV 热尾时必须扩大速度上界并同步
增加网格数。至少检查：热分布在外边界已足够小、网格收敛、粒子平衡、
电流符号随 `v_parallel` 共振侧反转，以及结果对 `dt` 收敛。

相对论计算设置 `coordinate_system='relativistic_momentum'`，并用
`u_parallel_max/u_perp_max` 控制无量纲动量上界。NPZ 输出此时保存
`u_parallel/u_perp`。相对论求解器把物理速度单位的 `Dql` 除以 `c^2`
映射到 u 空间；该近似还没有热介电张量与轨道坐标变换，必须做网格和功率矩收敛检查。

## 下一步物理闭合

真正预测 28 GHz O-mode 或 50 GHz X-mode/EBW 启动电流还需要：

1. 对实际 28/50 GHz 发射运行网格收敛的热等离子体全波求解，并逐时刻提供场图；
2. 将当前直接积分的相对论 Braams-Karney/Beliaev-Budker 参考后端加速为五势方程多重网格生产求解器；
3. 从 TPC/VFC 的真实 `B(R,Z,t)` 和场线/轨道替换当前镜像 bounce surrogate；
4. 在物理空间积分 `J_parallel B_phi/B`，并与电路和双极场闭合迭代；
5. CFS 形成后才切换到磁面/bounce-averaged 后端。
