# linEC 物理缺口与实现优先级（2026-10）

本审计面向两个最终输出：设备是否能完成 EC 启动，以及启动段 `Ip(t)` 大小。
当前代码可以做机制消融和参数敏感性，但质量门仍为 `diagnostic_only`。

机器可读审计入口：

```powershell
python -m ecstartup assess-p0 configs/linec_parameters.ini --evidence path/to/p0_evidence.json
python -m ecstartup audit-validation-manifest path/to/frozen_validation_manifest.json
```

审计使用文件存在性和 SHA-256，而不是只相信 `enabled` 开关。当前已接通主工作流的
增量包括事件化启动窗口、直接 X-B 的 R/L/UHR/倏逝层诊断，以及 1D-2V 对电离、
吸收、壁损失和电流矩的逐时刻替换反馈。它们仍需下述真实输入和独立验证才能完成 P0。

## P0：定量结论前必须完成

1. **真实 pre-CFS 外场与第一壁**
   - 输入 PF/TF/CS/TPC/VFC 线圈几何和逐炮电流波形，计算 `B(R,Z,t)`；
   - 用真实壁/限制器三角网格做场线命中、连接长度和 `B(s)`；
   - 由场线/导引中心轨道得到损失锥和快电子约束，替换常数连接长度及
     `tpc_mirror_ratio/tpc_trapped_fraction`。
2. **截面/EEDF 一致的击穿化学**
   - 用版本化 LXCat/AMJUEL/HYDHEL 数据替换解析反应率；
   - 加入 D2 振转激发、解离、D2+/D+/D3+、复合和辐射通道；
   - 电离、动量和能量损失率必须由同一个瞬时 EEDF 积分，不能继续分别拟合。
3. **功率归一化波-粒子耦合**
   - 由射线/全波给出空间和速度空间 `Dql(R,Z,u_parallel,u_perp,t)`；
   - 实现相对论热介电张量、偏振、折射、吸收和多次通过；
   - 同时满足射频功率守恒和动量输入守恒，禁止通过调整碰撞率匹配电流。
4. **空间动理学和轨道损失**
   - `open_field_line_1d2v` 已保留 `s,u_parallel,u_perp` 并可由主工作流选择；
   - F-P 分布已用于电离、壁通量、吸收和电流矩，并以替换方式反馈全局方程；
   - 在真实 `B(s,t)` 上 bounce-average，并显式处理 trapped/passing、损失锥和
     轨道不对称仍未完成；输入场线和 Dql 缺失时不会退回合成“真实值”。
5. **设备级验证集**
   - 每个装置都要包含成功炮和失败炮；
   - 同时比较 `Ip(t), ne(t), H-alpha/D-alpha, HXR/ECE, loop voltage`；
   - 报告启动分类灵敏度/特异度、Ip 误差和参数不确定度，并做留炮/留装置验证。

## P1：影响机制解释与阈值

1. **完整 Coulomb 碰撞算子**：当前场粒子项是线性化动量修正，不是完整
   Rosenbluth 势；需验证粒子、动量、能量守恒和 Maxwell-Juettner 平衡。
2. **鞘层和双极势空间结构**：当前为 0D Boltzmann 抑制，应改为沿场线电势、
   Bohm 边界、二次电子发射与壁材料依赖。
3. **动态电路**：用线圈-等离子体互感矩阵和实测 `dIcoil/dt` 求环电压；弱 CS
   不能仅按标签处理。
4. **壁/杂质后端**：用 ADAS ADF11/ADF15、材料相关溅射与再循环数据替换解析
   surrogate，并加入壁温、滞留和空间非均匀性。
5. **诊断前向模型**：从模型态计算干涉仪线积分、H-alpha 辐射、HXR/ECE 和磁探针
   信号，避免把体平均 `ne/Te` 直接与诊断量比较。

## P2：CFS 转换与 EBW

1. pre-CFS 阶段不求 GS；当场线扫描首次出现稳定闭合区域后，切换到自由边界
   GS/输运迭代，并检查磁能和电流剖面连续性。
2. O-X-B 需要全波/FEM 或经验证的 mode-conversion 模型：当前只能识别 O-cutoff、
   UHR 和几何共振候选，不能给 X-B 效率或 EBWCD。
3. CFS 后才能使用磁通坐标、真实通量面平均和闭合轨道；当前径向电流剖面只是
   对 0D 电流的归一化重建。

## 平衡文件的正确用途

- G-EQDSK/MAT 闭合平衡可提供装置尺度、LCFS/限制器形状、`R0/a`、伸长比、
  三角形度、体积和参考 Bt；不能替代 0--100 ms 的 pre-CFS 磁场历史。
- 本轮实测读取了：
  - J-TEXT `EQ_data_1109520_340beta0.3.mat`：`Bt=1.8694 T`、`Ip=78.11 kA`、
    `Raxis=1.0548 m`、等效 `a=0.2256 m`、`A=4.675`、`kappa~=0.994`；
  - WSL 中 EXL/XL LM 与 DN 参考平衡；
  - WSL 中 HL-2A `g037497.01200` 参考平衡。
- MAT 入口 `inspect-equilibrium-mat` 和 G-EQDSK 入口
  `tools/extract_gfile_parameters.py` 都显式输出“shape/reference only”边界。

## 建议实施顺序

1. EXL-50U 线圈波形 + 第一壁 + 场线/轨道损失；
2. 截面数据库和分子反应网络；
3. 空间功率归一化 Dql + 1D-2V F-P；
4. 用 EXL-50U 成功/失败炮完成第一轮设备标定；
5. 冻结模型后盲测 J-TEXT/HL-2A 等常规托卡马克，再扩展到其他球形环。
