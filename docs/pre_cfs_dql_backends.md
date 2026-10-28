# Pre-CFS Dql 生成与外部代码接入

## 物理结论

没有闭合磁面时仍可计算准线性扩散张量。`Dql` 的基本输入是局域波场谱、
磁场、密度、温度和回旋/Doppler 共振；闭合磁面只是一种方便的空间坐标和
bounce-average 基础。启动阶段应改用实验室坐标 `(R,Z,phi)` 或真实开场线
坐标 `s`，而不是把 500 ms EFIT 磁面外推到 0–100 ms。

仍然缺少逐时刻空间密度或发射边界条件时，不能唯一计算真实 `Dql`。代码可
生成参数化背景下的 `Dql` 用于机制/灵敏度分析，但必须保留输入假设和证据
等级。

## linEC 当前路径

1. `genray_like_internal`：由 `MagneticFieldProvider`、冷 Stix Hamilton 射线、
   相对论回旋共振核和吸收功率矩生成 0D-2V `Dql`。不需要 GS/EFIT，适合
   传统电磁 EC 候选机制；不能证明 X-B 或 EBW。
2. `project-fullwave-dql`：读取任意外部 FEM/FDTD 输出的实验室坐标复电场、
   波数、功率密度和端口功率预算，投影到开场线 1D-2V 网格。它不要求磁面。
3. 内部 `hot_fullwave.py`：局部 Nedelec/PML 热介质求解器，可做数值方法和
   局部转换层研究；尚不是包含真实端口和全装置边界的生产后端。
4. `synthetic_gaussian`：仅用于符号和灵敏度测试，不能报告 ECCD/EBWCD。

`python -m ecstartup assess-dql <config>` 会输出上述来源的机器可读质量门。

## 可接入的外部代码

- [GENRAY](https://github.com/compxco/genray) +
  [CQL3D](https://github.com/compxco/cql3d)：成熟的射线/准线性 F-P 工作流，
  适合已有平衡和磁通坐标的基准。公开仓库没有清晰 SPDX 许可证，因此本项目
  不复制其源码，只保留 NetCDF/NPZ 适配器。标准输出若基于闭合磁通面，不能
  未经坐标重建直接用于 pre-CFS。
- [Petra-M RF](https://github.com/piScope/Petra-M--RF) 与
  [Petra-M RFP](https://github.com/piScope/Petra-M--RFP)：GPL-3.0，支持 2D/3D
  频域 Maxwell、各向异性介质、端口和 PML；其任意空间场输出适合通过
  `FullWaveFieldMap2D` 合同接入。是否包含足够准确的热 EBW 介电模型和模式
  分解仍需针对具体版本验证。
- [FOCAL Benchmark](https://github.com/sterben168/FOCAL_Benchmark)：MIT，3D
  冷磁化等离子体 FDTD，允许外部密度剖面并输出 HDF5 电场。它适合验证 O/X
  传播、截止和转换附近电磁场，但冷模型本身不能给出热 Bernstein 波的定量
  `Dql`。

PIC 程序也能从波粒相互作用直接估计速度空间扩散，但把电子回旋时间尺度和
装置级 0–100 ms 同时解析通常不可承受，更适合局部转换层或降阶模型标定。

## 外部全波最小证据合同

除复数 `E_plus/E_minus/E_parallel`、`k_parallel`、`B` 和吸收功率密度外，
元数据必须包含：

```text
coordinate_basis = laboratory_RZ_* or open_field_line_s_*
requires_closed_flux_surfaces = false
power_normalized = true
wave_branch = electromagnetic_ec | ebw | mixed
mode_decomposition_verified = true
conversion_evidence = verified_fullwave
```

并满足端口功率守恒、局域吸收体积分、网格收敛和 PML/边界收敛。EBW 分支还
应保存高 `k_perp`、有限 `k_parallel`、近静电极化和热 Bernstein 色散残差。
随后分别构造 `Dql_direct` 与 `Dql_EBW`，用相同 F-P 初态做消融，才能区分
传统 ECCD、EBWCD 和混合驱动。

当前 MAST 示例可直接运行：

```powershell
python -m ecstartup generate-genray-like configs/devices/mast_ebw_startup.ini `
  --output runs/mast_pre_cfs_dql/internal_dql.npz `
  --summary runs/mast_pre_cfs_dql/internal_dql_summary.json
python -m ecstartup assess-dql configs/devices/mast_ebw_startup.ini `
  --dql-file runs/mast_pre_cfs_dql/internal_dql.npz
```

该结果有实验室坐标和功率归一化，且不使用闭合磁面；但由于内部冷射线没有
解决 X-B，本例有意输出 `electromagnetic_ec_or_unresolved_oxb`，而不是
`ebw`。
