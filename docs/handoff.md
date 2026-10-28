# 与已有 GENRAY/CQL3D/平衡代码的衔接

如果已有代码假定“磁面、密度/温度剖面和初始平衡已经存在”，它应放在 CFS 事件之后，不能直接拿来求击穿前阶段。推荐的 handoff 结构如下：

```text
startup (本目录)
  ├─ breakdown/preionization：开场线、非 Maxwellian、壁损失
  ├─ cfs_candidate/cfs：输出 ψ、O/X 点、LCFS、Ip、诊断时刻
  └─ handoff.nc
          ↓
equilibrium/GENRAY/CQL3D
  └─ 读取闭合磁面上的 ψ、ne(R,Z)、Te(R,Z)、B、RF launcher
```

CFS 前不应把 CQL3D 的磁面/bounce-averaged 坐标硬套到无闭合面的阶段。
当前项目因此提供物理速度空间的独立 `0D-2V` 后端。波程序向它交接的最小
合同是与速度网格同形且经过功率归一化的
`D_parallel_m2_s3/D_perp_m2_s3`；反向交接包括 `f_e`、`J_parallel`、压力矩
和守恒残差。CFS 后才把分布映射到磁面坐标并切换到完整 CQL3D 类后端。

## 建议的 handoff 变量

| 变量 | 单位 | 说明 |
|---|---|---|
| `t_cfs_s` | s | CFS 事件时间及不确定度 |
| `Ip_A` | A | CFS 时刻和后续时间序列 |
| `psi_Wb` | Wb/rad | 二维通量函数（v0.1 尚未生成真实二维场） |
| `R_axis_m`, `Z_axis_m` | m | 磁轴位置 |
| `R_lcfs_m`, `Z_lcfs_m` | m | LCFS 轮廓 |
| `ne_profile_m3`, `Te_profile_eV` | m⁻³, eV | 闭合面内剖面 |
| `I_ohmic/I_RF/I_tail/I_vertical_charge` | A | 启动阶段机制分解和不确定度；`I_orbit` 仅为旧别名 |
| `rf_frequency_Hz`, `k_parallel` | Hz, m⁻¹ | 波传播初值 |

在 v0.1 中，`state.nc` 只包含体平均量；正式 handoff 前必须完成真实线圈场、GS/场线追踪和空间剖面模块。任何把 `circular_surrogate` 的 `closed_fraction` 直接转换成 GENRAY 磁面输入的做法都应被质量门拒绝。
