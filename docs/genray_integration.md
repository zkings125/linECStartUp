# GENRAY 原生输入、内部降阶后端与验证

## 三种后端不能混称

linEC 现在把以下三类数据路径明确分开：

1. `inspect-genray` / `read_genray_netcdf` 读取原生 `genray.nc`。它导入射线路径、
   剩余功率、沉积剖面和 GENRAY 已算出的总电流，自动做 cgs 到 SI 转换。标准
   GENRAY NetCDF **没有** 0D-2V F-P 所需的速度空间 `D_ql`，因此不能把它直接
   填入 `rf_diffusion_file`。
2. `kinetic.rf_diffusion_model='genray_npz'` 读取用户或外部耦合程序导出的
   `D_parallel_m2_s3` / `D_perp_m2_s3` NPZ。这是项目自定义交换格式，不是
   GENRAY 原生 NetCDF。
3. `kinetic.rf_diffusion_model='genray_like_internal'` 用 linEC 自己的冷 Stix
   Hamilton 射线、相对论回旋谐波共振核、沿程降阶功率耗尽和 Maxwellian 功率矩
   归一化生成 `D_ql`。它不使用 CFS/GS/EFIT，但不是完整 GENRAY。

## 原生 NetCDF 入口

```powershell
python -m ecstartup inspect-genray `
  data/genray_reference/ec_iter_central_cd/gold-genray.nc `
  --output runs/genray_iter_regression/genray_native_summary.json `
  --plot runs/genray_iter_regression/genray_native_validation.png
```

转换规则包括：`cm -> m`、`gauss -> T`、`particles/cm3 -> m-3`、
`erg/s -> W`、`A/cm2 -> A/m2` 和 `erg/(cm3 s) -> W/m3`。射线数组严格按
`nrayelt` 截断。输出同时检查电子/离子/碰撞吸收功率和径向 bin 功率之和。

## 内部 Dql 生成

```powershell
python -m ecstartup generate-genray-like configs/linec_parameters.ini `
  --output runs/genray_like_exl50u/dql.npz `
  --summary runs/genray_like_exl50u/summary.json
```

对每个射线点求解共振核

\[
K=\exp\left[-\frac{(\omega-k_\parallel v_\parallel
-s\Omega_e/\gamma)^2}{2\Delta\omega^2}\right],
\qquad \gamma=(1-v^2/c^2)^{-1/2}.
\]

降阶吸收模型按 `dP/ds=-2 alpha P` 分配路径权重。随后以初始 Maxwellian 的
能量矩归一化 `D_ql` 幅度，使扩散算子的瞬时能量输入等于配置的参考吸收功率。
逐时刻双向 F-P 耦合仍按 `P_abs(t)/P_ref` 缩放这个参考张量。

这个实现现在可生成满足正半定约束的 `D_parallel/D_perp/D_cross`，并可接入
相对论动量网格和 pre-CFS 镜像 bounce surrogate；仍没有热/相对论介电张量、
波束多射线谱、完整波耗尽迭代、真实导引中心轨道或伴随方程。
O-X-B 的 X-B/EBW 段仍需热 Bernstein 色散或全波求解，不能用冷射线冒充。

## 官方 ITER 算例仅用于软件回归

公开输入来自 `compxco/genray` 的
`00_Genray_Regression_Tests/ci-tests/test-EC-ITER-Centra-CD`，固定上游 commit
记录在 `data/genray_reference/ec_iter_central_cd/upstream_commit.txt`。一键回归：

```powershell
python -m ecstartup validate-genray `
  configs/genray_iter_regression.ini `
  data/genray_reference/ec_iter_central_cd/gold-genray.nc `
  --output runs/genray_iter_regression
```

当前原生读取结果为：170 GHz、注入约 1 MW、吸收约 999.905 kW、平行电流
`-25.820 kA`、环向电流 `25.815 kA`；通道和径向功率守恒误差均低于 `1e-12`。

该 ITER 案例是已有平衡上的常规托卡马克中心 ECCD，不是 0--100 ms 启动，
更不是无 CS 启动验证。它只校验原生 NetCDF 字段、单位换算、功率守恒和电流
符号/数量级接口。启动物理报告应优先使用 EXL-50U 的 pre-CFS 输入；常规装置
困难机制可用 `configs/jtext_ec_startup.ini` 做可证伪的代表性诊断。

内部降阶模型的独立结果必须如实区分：

- 使用标准无量纲 ECCD 效率
  `zeta=e^3 n_e R I/(epsilon_0^2 P T_e)`、取显式输入 `zeta=0.30` 的伴随量级
  估算与 GENRAY 结果在同一量级；它不修改 `D_ql`，也不参与 F-P 反馈。
- 原始非相对论 0D-2V 瞬态 F-P 电流远高于 GENRAY。验证入口仍将其标为失败质量门，
  不通过暗调碰撞频率或几何面积把它拟合到参考值。新增相对论动量、线性化
  场粒子项和镜像 bounce surrogate 是必要但不充分的物理闭合，不能倒推成已通过 ITER。

因此，原生 NC 读取和功率守恒已经通过回归；内部射线到功率归一化 `D_ql` 的软件
链已经可运行；但“由内部 F-P 定量复现 GENRAY 电流”尚未通过，不能写成已验证。

## 性能

F-P 求解器缓存常系数情况下的稀疏矩阵和 SuperLU 分解。当前基准中，`41 x 28`
网格连续 40 步相对于每步重建求解器约加速 16 倍；时间变化的 `ne/Te/E/D_ql`
会自然产生新的缓存项。先保留这个可验证的算法优化，暂不引入 C++/pybind11，
避免在物理模型仍快速变化时增加 ABI 和构建复杂度。

## 文献边界

- Smirnov, Harvey & Prater, *General Linear RF-Current Drive Calculation in
  Toroidal Plasma*, DOI `10.1142/9789812814647_0042`。
- Lin-Liu et al., *Electron cyclotron current drive efficiency in general
  tokamak geometry*, DOI `10.1063/1.1610472`。
- Marushchenko et al., *Electron cyclotron current drive calculated for ITER
  conditions using different models*, DOI `10.1088/0029-5515/48/5/054002`。

这些论文支持热相对论吸收、伴随响应和模型间比较的必要性；本文档没有从无法核验
的全文中摘录具体电流值，官方数值仅来自随 GENRAY 仓库发布的 `gold-genray.nc`。
