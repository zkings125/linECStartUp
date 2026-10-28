# 跨装置 EC 启动算例

## 目的与边界

这一组算例用同一套 0D 启动方程比较不同装置显式参数，优先覆盖球形环/低纵横比装置，再加入常规托卡马克对照。它回答的是“在当前共同模型中，几何、场、气压、EC 功率和 CS 口径如何改变电离、电流与候选 CFS 判据”，不是已经通过实验验证的装置级预测。

每个案例使用其输入文件中的 `numerics` 积分范围和 `startup_window` 事件语义；不存在跨装置统一的 `0-100 ms` 假设。现有示例文件仍可能采用 100 ms 作为代表性数值窗口，但这不是物理定义。启动前不读取 GS/EFIT 闭合磁面；`closed_fraction` 仅为解析自场与外场竞争的候选事件。逐炮 PF/TF/CS/TPC/VFC 波形、真实第一壁、功率归一化 `Dql`、气压/Zeff 和成功/失败炮验证完成前，报告中的 `Ip` 必须称为 surrogate。

## 输入文件

输入位于 `configs/devices/`：

- `mast_ebw_startup.ini`：MAST 无螺线管 EBW 启动；
- `sunist_ec_startup.ini`：SUNIST 2.45 GHz 无感启动；
- `late_ech_startup.ini`：LATE ECH 低纵横比平衡形成；
- `quest_ec_startup.ini`：QUEST 反向环向场取消后的 EC 启动区间；
- `vest_tpc_startup.ini`：VEST TPC + 低环压 ECH 辅助启动；
- `diiid_ech_assisted_startup.ini`：DIII-D 低电压欧姆 + ECH 辅助启动；
- `jtext_no_cs_startup.ini`：J-TEXT 105 GHz X2 无 CS 代表性对照。

每个文件均为独立 INI。字符型值使用单引号；布尔开关使用 `switchOn = 'enabled'` 或 `'disabled'`。运行时真正参与扫描的功率来自 `[threshold_scan] ec_powers_W`，不是报告脚本中的硬编码列表。

## 可调参数

最常修改的段如下：

- `[geometry]`：`R0_m`、`minor_radius_m`、`B_t_T`、`B_vertical_T`、`B_penetration_T`、`connection_length_m` 和 `field_configuration`；
- `[gas]`：`pressure_Pa`、气体种类和碰撞数据版本；
- `[rf_source:*]`：频率、谐波、功率、吸收率、脉冲和发射几何；
- `[circuit]`：`cs_mode`、环电压、总电感和 Zeff；
- `[threshold_scan]`：功率、压强、Zeff 和 CS 扫描轴；
- `[validation]`：装置级预测质量门。默认必须保持 `intended_use = 'diagnostic_only'`。

文件头逐项区分了 DOI 直接约束、装置尺度参数和 `representative assumption`。不要为了得到期望启动结论而修改吸收率、连接长度或热尾不对称；这些量应由测量、真实场线或经验证的波/动理学后端替换。

## 运行

先校验单个输入：

```powershell
python -m ecstartup validate configs/devices/mast_ebw_startup.ini
```

运行全部装置、全部功率点并重建 PDF：

```powershell
python tools/build_cross_device_startup_report.py
```

## 输出

- `output/pdf/cross_device_ec_startup_report.pdf`：多页中文对比报告；
- `runs/cross_device_startup/results.json`：每个功率点的 `Ip(t)`、`ne(t)`、`Te(t)`、事件、分时段机制和归一化功率；
- `runs/cross_device_startup/resolved_inputs/*.json`：严格 schema 解析后的 SI 输入；
- `output/figures/cross_device_startup/*.png`：报告中的独立图件。

结果将“瞬时越过击穿密度”与要求密度、温度和净增长率持续满足的 `breakdown` 事件分开。峰值密度超过阈值但状态机仍停留在 vacuum 并不矛盾：前者是瞬时量，后者是带持续性和正净增长率的阶段判据。

## 当前结果应如何理解

第一轮统一模型在最高功率点给出的峰值电流约为：MAST `0.86 kA`、SUNIST `0.72 kA`、LATE `1.31 kA`、QUEST `5.15 kA`、VEST `16.5 kA`、DIII-D `23.2 kA`、J-TEXT `3.78 kA`。VEST 与 DIII-D 含 CS/环压，不能把总电流当作 EC 非感应电流；报告另列相对零功率基线的 EC 增量。

MAST 文献在 100 kW 下约为 17 kA，而当前统一模型的结果远低于该锚点。这一差距说明 MAST 的定量 O-X-B/EBW `Dql`、垂直场斜坡和轨道约束尚未进入跨装置全局闭合，不能用单个经验 `A/W` 常数掩盖。相同原因也限制了 QUEST、LATE 和 J-TEXT 的定量比较。

## 升级到定量预测

1. 接入逐炮线圈电流和真实壁几何，逐时刻计算场线命中、连接长度和轨道损失。
2. 对每台装置使用经过功率守恒、动量守恒和实验吸收验证的 `Dql(R,Z,u_parallel,u_perp,t)`。
3. 用实测气压、Zeff/杂质和正式原子数据库替换代表性输入。
4. 同时用成功炮与失败炮验证启动分类，并用独立 `Ip(t)`、`ne`、Halpha/Dalpha、HXR/ECE、loop voltage 检验机制。
5. 质量门通过后再比较装置级启动阈值和电流误差；在此之前只比较机制、趋势和敏感性。
