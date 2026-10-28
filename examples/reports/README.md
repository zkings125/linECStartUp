# Audited PDF reports

本目录只发布在正文中完整披露可信度边界的报告。它们用于展示模型输出、物理假设和仍未
通过的质量门，不代表 linEC 已经能够对任意装置给出炮级定量启动电流预测。

入选条件：

1. 明确区分软件回归、机制/灵敏度分析、文献锚点和实验验证；
2. 明确说明 pre-CFS 阶段没有使用 GS/EFIT 闭合磁面；
3. 披露代理模型、人工校准量和缺失的炮级输入；
4. 给出升级为定量预测所需的质量门或下一步输入；
5. 全部页面可解析，逐页渲染后无空白、裁切、重叠或不可读字符。

## 已发布报告

- `exl50u_advanced_kinetics_report.pdf`：EXL-50U 0--100 ms机制与灵敏度案例、相对论
  F-P数值验证和J-TEXT几何附录。正文明确说明尚缺炮级线圈波形、功率归一化Dql、
  壁/轨道损失及成功/失败炮联合验证。
- `jtext_representative_startup_report.pdf`：J-TEXT 105 GHz无CS代表性失败案例。
  它说明二次谐波几何可达并不足以保证启动，且不把热尾代理电流称为ECCD/EBWCD。
- `cross_device_ec_startup_report.pdf`：球形环优先的跨装置统一接口比较。所有Ip均标为
  surrogate，并在末页列出逐炮波形、第一壁、Dql和多诊断联合验证质量门。
- `mast_ebw_anchor_calibration_report.pdf`：使用文献电流锚点反演EBW动量假设和VFC斜坡的
  充分性测试。正文明确说明它不证明O-X-B真实发生，也不证明生成波形是MAST实测输入。

文件页数、大小、SHA-256和逐份证据说明见 `manifest.json`。

## 未发布的本地报告

- 旧版概览和重复EXL报告：可信度边界不如上述新版完整；
- `validation_diagnostics.pdf`：含尚待shot log确认的候选炮标签和未标定原始诊断量；
- `genray_validation_report.pdf`：主要是已有平衡上的ITER ECCD接口回归，不是启动案例；
- 其余临时阈值图：没有形成独立、完整的可信度说明。
