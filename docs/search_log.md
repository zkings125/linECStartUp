# 文献检索记录（可复现）

检索日期：2026-08-31。目标是为“跨装置 EC 启动机制”提供可核验的模型依据，而不是编制穷尽式综述。

## 第一轮：主题检索

- 数据源：Crossref works API（题名/摘要）、期刊 DOI 元数据；
- 关键词组合：`electron cyclotron startup tokamak`、`EBW current startup`、`open field line loss`、`spherical tokamak ECH`、`preionization`；
- 主题簇：气体击穿与 Townsend、开放场线/SOL、ECRH/ECCD/EBW、Grad–Shafranov/CFS。

## 第二轮：引用链/装置核验

逐一核对 LATE、MAST、SUNIST、QUEST、EXL-50、DIII-D、NSTX-U 的 DOI、作者、期刊、卷期页码和可用摘要。检索结果中只有 DOI 和出版社元数据被写入本仓库；未把搜索摘要当作超出摘要证据范围的定量结论。

## 证据分级规则

- **强**：多台装置或实验 + 独立诊断/射线或动理学模型相互支持；
- **中**：单台装置的机制诊断或数值研究，尚需跨装置复核；
- **弱/待定**：理论推测、单一拟合或缺少机制分项诊断。

完整映射见 [`literature.md`](literature.md)，BibTeX 见 [`references.bib`](references.bib)。下一步若用于论文，应在 Web of Science/Scopus/CNKI 补充系统检索、去重、引用网络和中文文献核验，并把数据库访问日期与检索式存档。
