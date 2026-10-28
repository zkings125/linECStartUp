# 文献证据与跨装置建模映射

本页按“第一轮基础机制 → 第二轮装置证据”组织。文献只用于提出参数和可证伪判据；不同装置的功率、频率、气压、垂直场和诊断定义不同，不能直接比较一个未经归一化的 A/W。

## 第一轮：方程和数值基础

| 主题 | 推荐资料 | 在代码中的落点 |
|---|---|---|
| Townsend/Paschen、鞘层、气体放电 | Y. P. Raizer, *Gas Discharge Physics*；M. A. Lieberman & A. J. Lichtenberg, *Principles of Plasma Discharges and Materials Processing* | `K_ion(E/N)`、二次发射、`τ_loss`、Bohm 边界 |
| 开放场线/SOL/壁相互作用 | P. C. Stangeby, *The Plasma Boundary of Magnetic Fusion Devices* | `L_conn`、再循环、壁通量和损失锥 |
| 磁平衡与 CFS | J. P. Freidberg, *Plasma Physics and Fusion Energy*；J. Wesson, *Tokamaks* | `ψ(R,Z)`、GS 接口、O/X 点和 LCFS 判据 |
| 碰撞输运与诊断 | H. Helander & D. J. Sigmar, *Collisional Transport in Magnetized Plasmas*；I. H. Hutchinson, *Principles of Plasma Diagnostics* | `ν_en/ν_ei`、磁化参数、实验前向模型 |
| 电子输运/截面 | LXCat、BOLSIG+、LoKI-B、AMJUEL（使用时记录版本） | `RateProvider`，替换 `analytic_surrogate` |

## 第二轮：与本问题直接相关的装置证据

### 球形/低纵横比装置

0. **VEST/KSTAR：TPC 位形本身是磁场配置**

   Y. An, J. Lee, J. Jo *et al.*, “Efficient ECH-assisted plasma start-up using trapped particle configuration in the versatile experiment spherical torus,” *Nuclear Fusion* **57** (2017) 016001. DOI: [10.1088/0029-5515/57/1/016001](https://doi.org/10.1088/0029-5515/57/1/016001)。文中把 TPC 明确定义为 mirror-like magnetic field configuration：在环向电压开启前形成，利用镜像效应提高预电离粒子约束，并提供较稳定的平衡场。该 TPC 是位形层概念，不应与 EXL-50 论文中的 `j_TPC` 缩写混写。

   J. Lee, Y. An *et al.*, “Study on ECH-assisted start-up using trapped particle configuration in KSTAR and application to ITER,” *Nuclear Fusion* **57** (2017) 066001. DOI: [10.1088/1741-4326/aa8511](https://doi.org/10.1088/1741-4326/aa8511)。该研究比较 TPC 与 conventional field-null configuration，报告 TPC 对预电离质量、压力窗口和低环向电场启动的影响。

   **模型映射**：`field_configuration="TPC"` 只选择镜像场 provider；镜像比、捕获比例、衰减指数和有效连接长度必须与线圈/场线数据分别标定。常规垂直场案例使用 `field_configuration="VFC"`，不要把两种配置压缩为一个 `B_vertical` 标量。

1. **SUNIST：瞬态过程与气压控制**

   Y. Tan, Z. Gao, L. Wang *et al.*, “Transient process of a spherical tokamak plasma startup by electron cyclotron waves,” *Nuclear Fusion* **51** (2011) 063021. DOI: [10.1088/0029-5515/51/6/063021](https://doi.org/10.1088/0029-5515/51/6/063021)。摘要将瞬态过程归因于电离、开场线损失和梯度-B 漂移的组合，并指出低频微波启动对填充气压很敏感。

   **模型映射**：`pressure_Pa`、`L_conn`、`orbit_asymmetry`、`K_ion`；实验判据是微波反射、Hα/Dα、垂直场扫描和电流时序。

2. **LATE：ECH 形成球形托卡马克平衡**

   T. Maekawa, Y. Terumichi, H. Tanaka *et al.*, “Formation of spherical tokamak equilibria by ECH in the LATE device,” *Nuclear Fusion* **45** (2005) 1439–1445. DOI: [10.1088/0029-5515/45/11/026](https://doi.org/10.1088/0029-5515/45/11/026)。该类实验展示了低纵横比装置中用 ECH/ECCD 形成平衡的可行性。

   **模型映射**：CFS 形成后才切换到 GS/通量面模型；不能把形成后的 ECCD 直接外推到击穿前。

3. **MAST：EBW 非感应电流启动**

   V. F. Shevchenko, M. R. O’Brien, D. Taylor & A. N. Saveliev, “Electron Bernstein wave assisted plasma current start-up in MAST,” *Nuclear Fusion* **50** (2010) 022004. DOI: [10.1088/0029-5515/50/2/022004](https://doi.org/10.1088/0029-5515/50/2/022004)。摘要报告 100 kW RF 下约 17 kA、优化垂直场斜坡下约 33 kA 的无螺线管启动，并指出结果与射线追踪和准线性 Fokker–Planck 模型一致。

   **模型映射**：`I_RF`、`B_vertical` 波形、`rf_accessibility`、射线/2V 接口；这是直接波粒动量输入的强证据，但不等于所有 ST 的唯一机制。

4. **EXL-50：热尾优先约束与垂直电荷分离**

   M. Wang, D. Guo, Y. Shi *et al.*, “Experimental study of non-inductive current start-up using electron cyclotron wave on EXL-50 spherical torus,” *Plasma Physics and Controlled Fusion* **64** (2022) 075006. DOI: [10.1088/1361-6587/ac70fb](https://doi.org/10.1088/1361-6587/ac70fb)。作者用限制器探针区分 collisionless tail preferential confinement current (`j_TPC`) 与 vertical charge-separation current (`j_VTF`)，报告启动期间前者增加、后者减小；同时指出气压与垂直场乘积存在启动上限，只有足够低的气压和垂直场才容易启动。

   **模型映射**：必须保留规范机制名 `I_tail`（旧输出名 `I_orbit`/`I_tpc`）以及
   `I_vertical_charge`（旧输出名 `I_vfc`/`I_vtf`），并与 `I_RF` 分项；扫描
   `p×B_vertical` 时要把位形参数和电流闭合参数分开。EXL-50 的 `j_TPC`、
   `j_VTF` 是其限制器诊断中的电流密度/机制记号，不是本代码的 TPC/VFC
   provider 名称，也不与 `I_*` 逐点数值相等。限制器探针和壁电流是关键验证。

5. **EXL-50：微波预电离**

   B. Chen, Y. Zhu, Q. Zhou *et al.*, “Microwave preionization and electron cyclotron resonance plasma current startup in the EXL-50 spherical tokamak,” *Plasma Science and Technology* **24** (2022) 015104. DOI: [10.1088/2058-6272/ac3640](https://doi.org/10.1088/2058-6272/ac3640)。摘要报告 2.45 GHz 预电离可消除击穿延迟，并改善放电长度、稳定性和驱动电流；还比较了 2.45 GHz 与 28 GHz ECR 源。

   **模型映射**：把 `P_ECRH` 分为预电离/热尾/直接电流三条通道，输出击穿延迟而非只输出峰值电流。

6. **QUEST：反向环向电场抑制/释放热尾**

   T. Onchi, H. Idei, K. Hanada *et al.*, “Electron cyclotron current start-up using a retarding electric field in the QUEST spherical tokamak,” *Nuclear Fusion* **64** (2024) 106020. DOI: [10.1088/1741-4326/ad6914](https://doi.org/10.1088/1741-4326/ad6914)。摘要报告反向环向电场可抑制能量电子增长；取消该场后等离子体电流和硬 X 射线迅速增加，并观察到高密度/高温时第二谐波单程吸收增强。

   **模型映射**：需要 `E_loop` 极性、`E/E_D`、HXR/热尾状态和时间依赖波传播；说明“环向场只负责欧姆电流”是不够的。

### 常规高纵横比装置与“困难但并非绝对不可能”

7. **DIII-D 低电压欧姆 + ECH 辅助启动**

   B. Lloyd, G. L. Jackson, T. S. Taylor *et al.*, “Low voltage Ohmic and electron cyclotron heating assisted startup in DIII-D,” *Nuclear Fusion* **31** (1991) 2031–2053. DOI: [10.1088/0029-5515/31/11/001](https://doi.org/10.1088/0029-5515/31/11/001)。这类工作表明常规托卡马克可以用 ECH 辅助降低启动要求，但“辅助击穿/加热”与“纯 EC 非感应电流启动”是不同问题。

8. **Tore Supra / ITER 的 ECRH-assisted startup 研究**

   B. Lloyd, P. G. Carolan & C. D. Warrick, “ECRH-assisted start-up in ITER,” *Plasma Physics and Controlled Fusion* **38** (1996) 1627–1643. DOI: [10.1088/0741-3335/38/9/007](https://doi.org/10.1088/0741-3335/38/9/007)。该方向强调频率、波可达性、回路电压和装置几何的联合约束，支持把常规装置作为参数空间中的“困难区域”而不是绝对禁区。

9. **NSTX-U 的时变波传播/EBW 可行性**

   N. A. Lopez & F. M. Poli, “Regarding the optimization of O1-mode ECRH and the feasibility of EBW startup on NSTX-U,” *Plasma Physics and Controlled Fusion* **60** (2018) 065007. DOI: [10.1088/1361-6587/aabaa8](https://doi.org/10.1088/1361-6587/aabaa8)。摘要强调密度剖面演化决定 ECRH 可用时间窗，静态剖面不能捕捉 cutoff 附近的优化；并讨论 EBW 对早期电流和闭合磁面的潜力。

   **模型映射**：跨装置比较必须使用 `n_e(t)/n_critical` 和时变射线/吸收，而不是一个静态吸收率。

## 证据强度与待验证争议

| 结论 | 证据强度 | 当前代码态度 |
|---|---|---|
| EC 可在低纵横比装置中参与非感应电流启动 | 强（多装置实验） | 保留 `I_RF`、`I_tail` 和 `I_vertical_charge` 三条可审计通道 |
| 开场线损失/轨道不对称对早期净电流重要 | 中—强（SUNIST、EXL-50 诊断） | 显式 `L_conn`、`orbit_asymmetry`，待 1D/2V 替换 |
| EBW/ECCD 可在 CFS 形成后主导电流 | 强（MAST、LATE、QUEST） | CFS 前后分阶段，禁止混用闭合面公式 |
| ST 的优势主要来自某一个单独机制 | 弱/未定 | 用消融和无量纲矩阵检验，不预设答案 |
| 常规托卡马克绝对不能 EC 启动 | 不成立（存在 ECH-assisted startup） | 比较“辅助启动”与“纯非感应启动”的不同指标 |
