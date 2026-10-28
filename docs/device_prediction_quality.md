# 设备级定量预测质量门

`linEC` 将“几何/机理诊断”和“准确判断装置能否启动、Ip 多大”分开。
前者可在输入不完整时运行；后者必须通过 `[validation]` 质量门。

```powershell
python -m ecstartup assess-device-readiness configs/linec_parameters.ini
```

质量门要求炮级 PF/TF/CS/TPC/VFC 波形、实测气压、Zeff/杂质、标定 EC
光束与功率归一化 Dql、壁/限制器几何和轨道损失标定。还必须同时用成功炮和
失败炮验证启动分类灵敏度/特异度，并用独立 Ip(t) 轨迹验证相对误差。

默认 `intended_use = 'diagnostic_only'`。此时共振层、击穿/预电离阶段、机制分解、
参数敏感性、阈值括号和 surrogate Ip 可以输出，但不得称为设备级预测。
`intended_use = 'quantitative_device_prediction'` 本身不会放行；数据数量和验证指标
也必须达到配置阈值。

## 环径比扫描

`R0_m` 和 `minor_radius_m` 在 `[geometry]` 中本来就是两个独立输入。批量研究用：

```ini
[aspect_ratio_scan]
enabled = 'enabled'
R0_values_m = [0.45, 0.50, 1.00]
minor_radius_values_m = [0.20, 0.30]
combination = 'cartesian'
scale_rf_launch_with_geometry = 'enabled'
```

```powershell
python -m ecstartup aspect-ratio-scan configs/linec_parameters.ini `
  --output runs/aspect_ratio_scan.json
```

每个扫描点输出纵横比 `A=R0/a`、逆纵横比、热比压代理、真空环向场模型下可达
基频/二次谐波层、启动阶段、峰值 Ip 和机制诊断。这里没有把“ST 更易启动”写成
经验乘子；低纵横比的影响必须通过共振层、场线约束、损失、吸收和 F-P 分布反馈
产生。跨装置的每个真实装置仍需独立通过质量门。
