# EC 电流阈值扫描

## 阈值定义

`threshold-scan` 对每个 `(pressure, Z_eff, CS mode)` 先计算一个
`P_EC=0` 基线，再对配置的 EC 功率逐点运行同一套启动方程。EC 增量定义为

\[
\Delta I_{EC,peak}=\max_t\left[s_I\left(I_p(t;P_{EC})-I_p(t;0)\right)\right],
\]

其中 `s_I=current_direction_sign`。成功点必须满足：

- `Delta I_EC,peak >= success_increment_A`；
- 有符号总电流峰值不低于 `success_total_current_A`；
- 若 `require_preionization='enabled'`，还必须达到预电离密度和温度判据；
- ODE 求解成功。

关闭自适应细化时，报告值只是配置功率网格中最低的成功点。启用
`refine_enabled='enabled'` 后，程序在相邻失败/成功功率之间二分，直到区间宽度
不大于 `refine_power_tolerance_W` 或达到 `refine_max_iterations`。JSON 同时保存
`last_failed_ec_power_W` 和 `minimum_successful_ec_power_W`；后者是成功上界，
不是已证明的物理分岔点。

## 扫描量如何进入方程

- `pressures_Pa`：覆盖恒定注气目标，改变中性密度、`E/N`、电离和电子-中性碰撞；
- `effective_charges`：覆盖电路使用的 `Z_eff`，令 `nu_ei` 按 `Z_eff` 缩放；
- `ec_powers_W`：改变选定独立 EC 源的注入功率；
- `cs_modes='with_cs'/'without_cs'`：保留规定环电压或严格置零；
- `disable_other_sources='enabled'`：扫描一个源时关闭其他 EC 源，避免功率归因混乱。
- `kinetic_coupling_mode`：允许阈值大扫描显式选用响应表或逐时刻双向F-P；
  `inherit`沿用主配置。正式报告中的阈值必须连同该模式一起解释。

EXL-50U模板使用2、4、5、8、10 mPa（配置内SI值为0.002--0.010 Pa）和
70、80、125、220、300 kW。3 A弱CS按无CS分类；规定5 V环电压只作为独立
强CS对照，不能由3 A直接推得。

正常运行时，若 `[threshold_scan] switchOn='enabled'` 且
`[outputs] write_pdf_report='enabled'`，扫描曲线和阈值热图追加到同一个
`startup_report.pdf`，同时写出 `threshold_scan.json`。

也可独立执行：

```powershell
python -m ecstartup threshold-scan configs/linec_parameters.ini `
  --output runs/threshold_scan.json `
  --pdf-output runs/threshold_scan_report.pdf
```

## 当前可信度边界

内置 RF 吸收、热尾和 synthetic `D_ql` 仍是降阶闭合。即使启用F-P矩反馈，
只有把波迹/全波得到的功率沉积和功率归一化 `D_ql` 接入，并用真实
TPC/VFC 场线损失校准后，
阈值才可能用于装置预测。当前扫描用于识别模型内的趋势、耦合和待标定区间。
