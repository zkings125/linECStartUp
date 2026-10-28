# 从 0 到可发表结果的实施计划

## Step 0：先固定科学问题

不要先写求解器。先把三个事件写成可复现的观测定义：

```text
breakdown      Γ_net > 0 且 n_e 超过阈值并持续 N 个采样点
preionization  n_e、Te 和 Γ_net 同时满足持续条件
CFS            ψ 闭合区、场线连接长度和 O/X 点条件同时满足
```

同时定义“EC 启动成功”的等级：仅击穿、维持预电离、形成 CFS、形成可移交的等离子体电流。这样不同装置的论文不会把不同事件混成一个标签。

## Step 1：建立数据契约和单位

先实现 `GlobalState`、`SimulationConfig`、`Event`、`RateProvider/RFProvider/TopologyProvider`，写配置拒绝测试和单位测试。任何模块不得直接访问另一个模块的私有数组。

## Step 2：0D 基线和参数窗口

实现或接入：

1. 中性气体 `n_n=p/(kT)`；
2. 电子/离子/中性粒子守恒；
3. 电子能量和辐射/壁损失；
4. Faraday 回路和碰撞电导；
5. ECRH 处方功率、热尾矩和机制分解。

验收：Paschen/Townsend 极限、固定电导 L-R 阶跃、粒子/能量正性和 `dt` 收敛都通过。此阶段只用于回答“有没有窗口、哪个量最敏感”，不能宣称装置预测。

## Step 3：1D 开放场线（真正的击穿核心）

以场线坐标 `s∈[0,L_conn]` 解：

\[
\partial_t n_s+\partial_s\Gamma_s=S_s,
\quad
\Gamma_e=-\mu_en_eE_\parallel-D_e\partial_sn_e,
\]

加上电子能量、Poisson/准中性切换、Bohm 鞘层、二次发射、中性再循环和横向漂移损失。使用守恒有限体积 + IMEX/BDF，输出壁通量和局部 EEDF 需要的边界条件。

验收：单场线解析解、鞘层通量、网格/时间收敛；与 0D 在均匀极限下重合。

## Step 4：0D-2V/1D-2V 动力学

把 Maxwellian 假设替换为

\[
\partial_tf+v_\parallel\partial_sf+
\left[-\frac{eE_\parallel}{m_e}-\frac{\mu}{m_e}\partial_sB\right]\partial_{v_\parallel}f
=C_{en}+C_{ei}+C_{ee}+Q_{RF}+Q_{ion}.
\]

先做空间均匀 0D-2V，再接 1D 场线。由 `f` 直接计算 `K_ion`、迁移率、能量损失、热尾电流和损失锥通量；所有碰撞算子做粒子/能量守恒测试。

## Step 5：真实 RF 波传播

输入频率、极化、发射位置和 `k_parallel`，逐步替换：

```text
处方沉积 → 冷等离子体射线 → 热射线/模式转换 → 全波 + 准线性扩散
```

每一步都输出共振位置、反射/透射、吸收功率和动量源，避免用一个不可解释的 `A/W` 吸收所有误差。

## Step 6：磁场和 CFS 拓扑

先用线圈 Biot–Savart + 场线追踪，再接自由边界 GS。CFS 事件前禁止通量面平均；事件后才切换到 `ψ`、`q(ψ)`、新古典输运和 GENRAY/CQL3D 的闭合面接口。

验收：解析圆柱/Solov'ev 平衡、`div B`、O/X 点和 LCFS 连通性；拓扑事件对网格/场线步长收敛。

## Step 7：跨装置因果比较

对每个装置只替换已测量输入，统一输出：

```text
R0/a, a/R0, Bp/Bt, ne/ncritical, omega_ce/omega_RF,
L_conn, Da_ion, omega_c/nu, E/E_D, rho_e/a,
  I_ohmic, I_RF, I_tail, I_vertical_charge, t_breakdown, t_CFS
```

这里的 `I_tail` 和 `I_vertical_charge` 是代码的规范机制名；`I_orbit`、
`I_tpc`、`I_vfc`、`I_vtf` 仅为兼容别名。不要把它们与 EXL-50 论文中的
`j_TPC`/`j_VTF` 直接等同：论文的 `j` 是探针/局部电流密度记号，需经过
壁和限制器几何前向模型才能与代码的体积分电流（A）比较。

然后执行：

1. 同一装置的单参数扫描；
2. 四组机制消融；
3. 留一装置验证；
4. 参数不确定度 ensemble 和启动概率；
5. 与 `n_e`、Hα/Dα、HXR、壁探针、磁探针同步对齐。

只有当“模型能同时解释多个诊断且机制分项可识别”时，才可讨论 ST 与常规装置的因果差异。
