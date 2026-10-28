# 跨装置研究问题：为什么球形托卡马克更容易被 EC 启动？

这份代码最有价值的用途，是把“ST 容易、常规托卡马克困难”拆成可证伪的机制，而不是给两类装置贴一个经验标签。比较时必须保持反应截面、壁模型和事件定义一致，只改变已测量的机器参数。

## 需要区分的机制

### 磁场位形与电流机制不是同一维度

跨装置矩阵应至少把下面两列分开记录：

| 层次 | 代表量 | 物理问题 |
|---|---|---|
| 磁场位形 | `field_configuration = TPC/VFC`、`B_R/B_Z/B_phi`、`M`、`f_trap`、`L_conn` | 场线是否镜像/捕获、粒子如何到壁、垂直场如何随时间和半径变化 |
| 电流机制 | `I_RF`、`I_tail`、`I_vertical_charge` | 波粒动量、热尾优先约束和垂直电荷分离分别贡献多少环向电流 |

VEST 的 TPC 文献（An *et al.*, DOI `10.1088/0029-5515/57/1/016001`）把 TPC 描述为 mirror-like magnetic field configuration；KSTAR/TST-2 的后续研究比较的是 TPC 与 field-null configuration。不要把这里的 TPC 位形名直接等同于 EXL-50 的 `j_TPC`。后者是 collisionless tail preferential-confinement current 的文献诊断记号，不是磁场 provider。EXL-50 的 `j_VTF` 是 vertical charge-separation current，也不是 VFC 位形的定义；代码中的 `I_vtf_A` 只是该机制的兼容别名。论文中的 `j`（电流密度/探针量）与代码中的 `I`（体积分电流，A）还需要壁/限制器几何前向模型才能比较。

| 假设 | 代码中的可扫描量 | 若假设成立，应看到的趋势 | 关键反证/诊断 |
|---|---|---|---|
| H1：EC 波可达性/模式转换 | `B_t`、频率、`n_e/n_critical`、发射几何、`absorption_fraction` | ST 的低场/低密度窗口更宽，或 EBW 转换后吸收更强 | 反射/透射、X/O 模式、局域 HXR 与吸收位置 |
| H2：开场线损失较小或损失不对称更有利 | `L_conn`、壁几何、`secondary_emission`、`orbit_asymmetry` | 在相同 `E/N` 下 ST 的 `Da_ion` 和 `I_tail` 更大 | 限制器探针、壁电流、场线连接长度、电子能谱 |
| H3：极向场建立和外部穿透场竞争不同 | `R0/a`、`I_p`、`B_vertical`、线圈波形 | `B_p/B_t` 或 CFS 候选 margin 更快跨过阈值 | 磁探针/EFIT、O/X 点和 LCFS 时序 |
| H4：高能尾/直接波粒动量输入占主导 | `hot_fraction`、`I_RF`、共振几何、`E/E_D` | 电流分项随 RF 极性、`N_parallel` 显著改变 | HXR、ECCD 极性扫描、速度分布测量 |
| H5：CFS 形成后的反馈才是关键 | CFS 拓扑、闭合区持续时间、输运切换 | 事件前后 `τ_loss`、电导和 `I_p` 增长率出现跃迁 | 高频磁探针、干涉仪、回路电压和电流联合时序 |

## 建议的比较指标

不要只比较 `A/W`。每个装置至少输出：

```text
A = R0/a                         # aspect ratio
epsilon = a/R0                  # inverse aspect ratio
omega_ce/omega_RF                # 共振位置的场频比
ne/n_critical                    # EC 波可达性（同时记录 RF 开启时的 active 值）
rho_e/a                          # 导引中心近似
omega_c/nu                       # 磁化程度
lambda_D/a, lambda_mfp/a         # 准中性和流体近似适用性
Da_ion = nu_ion*tau_loss         # 电离对抗开场线损失（报告峰值和 active 值）
Bp_edge/Bt                      # 极向/环向场竞争
E/E_D                            # runaway/热尾预警
I_ohmic, I_RF, I_tail, I_vertical_charge  # 机制分解
closed_fraction, CFS time        # 拓扑跃迁
```

运行示例：

```powershell
python -m ecstartup compare configs/device_matrix.json --output runs/device_comparison.json
```

当前比较使用 `global0d_surrogate`，且 `device_matrix.json` 中的 ST_like/conventional_like 是方法演示，不是实际机器参数。因此结果只能用来检查“哪些无量纲组合值得实验验证”。装置结论必须在 1D 开场线、真实线圈场、截面/EEDF 和自由边界 GS 版本上复核。

真实装置数据请先填入 [`configs/device_catalog_template.csv`](../configs/device_catalog_template.csv)，每个数值必须附 DOI、实验记录或机器数据库来源；模板中的空值是有意保留的，不能用示例装置参数代替。

若要把“容易启动”定义为概率而不是单一阈值，可用 `ensemble` 命令传播气压、垂直场和 RF 功率的不确定度：

```powershell
python -m ecstartup ensemble configs/toy_all_stages.yaml `
  --parameters configs/ensemble_ranges.json --samples 32 --event preionization
```

输出包含 Wilson 95% 区间。这个概率是模型先验下的条件概率，不是装置可靠性指标；截面、壁面和测量误差必须纳入同一个 ensemble 才有解释力。

## 从比较到因果判定的实验设计

1. 在同一装置上独立扫描 `B_vertical`、气压、RF 功率和 `N_parallel`，先把机器间差异转成参数间因果曲线。
2. 做四组消融：仅环向电场、仅 RF 加热、仅轨道不对称、联合驱动；保留每项 `I` 和能量贡献。
3. 以磁拓扑事件为时间零点，对齐 `n_e`、Hα/Dα、HXR、Rogowski 和壁电流，而不是用触发时刻代替 CFS。
4. 用留一装置验证和参数后验预测启动概率，报告置信区间；若只能拟合单台机器，应明确标记为不可外推。

代码提供了一个最小消融命令：

```powershell
python -m ecstartup ablate configs/toy_all_stages.yaml --output runs/ablation.json
```

它分别关闭 RF、环向场、轨道不对称、直接 RF 电流和热尾，比较事件序列及电流分项。消融不是因果证明，但能快速发现“某一项被另一项完全代偿”的不可识别情形。
