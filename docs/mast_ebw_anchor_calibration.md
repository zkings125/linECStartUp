# MAST EBW 动量与 VFC 斜坡文献锚点校准

## 目的与边界

本工具回答一个受限问题：如果向现有启动模型提供足够强的 EBW 波粒平行动量输入和一个随时间增强的垂直场约束，模型是否能达到 Shevchenko 等人在 MAST 上报告的 `100 kW -> ~17 kA` 和优化垂直场斜坡后的 `~33 kA`。

答案不能解释为第一性原理预测，也不能据此断言 O-X-B 模式转换真的发生。当前公开可核验材料给出了电流锚点，但没有提供可直接导入本代码的炮级 `Bv(t)`、EBW `k_parallel` 谱或功率归一化 `Dql`。因此本模块反演两个明确命名、默认不进入普通算例的参数；报告中的机制名称为“EBW 动量假设”。

## 输入模型

基础 EBW 波粒动量写为

```text
I_EBW,base(t) = eta_EBW * P_abs(t)
```

`eta_EBW` 由恒定垂直场下的 17 kA 锚点反演。优化案例保持 `eta_EBW` 不变，使用

```text
G_ramp(t) = 1 + (G_max - 1) * progress(|Bv(t)|)^p
I_EBW(t) = I_EBW,base(t) * G_ramp(t)
```

其中 `G_max` 由 33 kA 锚点反演。`Bv(t)` 是 6 mT 到 14 mT 的生成波形，表示垂直场随电流爬升而增强的测试输入。它不是恢复的 MAST 线圈电流波形。

## 运行

从项目根目录执行：

```powershell
python tools/calibrate_mast_ebw_anchor.py --config configs/calibration/mast_ebw_anchor.ini
```

用户可编辑参数文件中的数值。所有字符串值必须放在单引号内，开关统一写作 `switchOn = 'enabled'` 或 `switchOn = 'disabled'`。

## 输出

- `runs/mast_ebw_anchor_calibration/results.json`：校准系数、锚点误差、功率扫描和分时段机制。
- `runs/mast_ebw_anchor_calibration/generated_ebw_vfc_inputs.csv`：逐时刻 EBW 动量输入和 VFC 波形。
- `runs/mast_ebw_anchor_calibration/generated_ebw_vfc_inputs.npz`：相同数据的 NumPy 压缩格式。
- `output/pdf/mast_ebw_anchor_calibration_report.pdf`：图像、参数、机制和限制说明。

## 替换为真实输入

未来获得实测 VFC/平衡场线圈电流后，应先通过线圈几何计算 `Bv(R,Z,t)`，再替换参数文件中的生成波形。获得全波或 GENRAY/CQL3D 输出后，应以速度空间动量矩

```text
integral p_parallel * Q_QL[f] d^3p
```

替换 `eta_EBW * P_abs`，并同时验证注入功率、吸收功率、电子能量和总平行动量守恒。只有在独立成功炮和失败炮上通过 `Ip(t)` 验证，才可去掉“文献锚点充分性测试”标签。
