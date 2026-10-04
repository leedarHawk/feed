# 基本面数据 v1（2026-10-04 从聚宽拉取）

数据源：**只有聚宽**（`finance.STK_*` 表、`get_fundamentals`、`get_valuation`），没有混用其他来源。
用途：价值、质量、成长因子，业绩预告漂移，筹码，解禁。区间：每日估值 2017-01-03 至 2026-09-30；财报、预告、股东户数、解禁按**公告日**覆盖 2015-01-01 至 2026-09-30（多拉两年，用于 2017 年起的滚动 12 个月与同比）。
格式：Parquet。聚宽原始下载（csv.gz）和拉取、整理脚本未随本仓库发布。

> **防前视的用法**：财报类数据一律按 `pub_date`（公告日）使用，不要按报告期 `end_date` 对齐。2023 年报 5,131 份里 4,052 份在 2024 年 4 月、1,058 份在 3 月公布。

| 文件 | 行数 | 说明 |
|---|---|---|
| `valuation_daily_<年>` | 合计 10,350,577 | 每日估值，5,469 只，2017-01-03 至 2026-09-30；**按年拆成 `valuation_daily_2017.parquet` … `valuation_daily_2026.parquet` 共 10 个文件**（整表 291MB，超过 GitHub 单文件 100MB 上限），各文件按 (code, date) 排序，合计行数与整表一致 |
| `financials_pit` | 193,545 | **季度财务摘要（带公告日）**，三表 + 指标对齐，推荐入口 |
| `income` / `balance` / `cashflow` | 412,356 / 412,377 / 412,361 | 三大报表完整科目（67 / 124 / 90 列），带公告日 |
| `indicator` | 200,716 | 聚宽财务指标，**单季度口径** |
| `forecast` | 79,264 | 业绩预告 |
| `holder_num` | 273,120 | 股东户数 |
| `unlimit_events` | 24,090 | 限售股实际解禁 |
| `unlimit_schedule` | 41,417 | 限售股解禁日程（含预计解禁日） |

代码是聚宽格式（`000001.XSHE`）。股票范围是 2015 年起任一时点存在过的 A 股（5,480 个代码，含已退市），代码清单未单独发布。

## 1. 每日估值 `valuation_daily_<年>`
`code, date, pe_ratio, pe_ratio_lyr, pb_ratio, ps_ratio, pcf_ratio, market_cap, circulating_market_cap, capitalization, circulating_cap, turnover_ratio`
- `pe_ratio` 滚动 12 个月（TTM），`pe_ratio_lyr` 按上一年报。市值单位**亿元**（平安银行 2026-09-30 为 2,245 亿，与实际相符）；股本单位聚宽文档为万股，我没有核对；换手率 %。
- **亏损、负净资产、负现金流的股票给负数，不是空值**：`pe_ratio<0` 占 19.5%，`pb_ratio<0` 占 0.8%，`pcf_ratio<0` 占 46%（最后一项偏高，我没有核实原因）。极端值很大（`pe_ratio` 从约 −373 万到 265 万），使用前需自行截尾或剔除负值。
- **公告日对齐（抽查）**：用平安银行 2023–2024 年倒推隐含利润（市值 ÷ 市盈率），它在每期财报公告当天（或周末公告后的下一个交易日）才更新，与 `pub_date` 吻合。2023-01-17 的一次更新早于 2022 年报公告日（2023-03-09），推测是业绩快报，该日数值已等于年报最终值，同样是当时已公开的信息。**只抽查了这一只股票**，不能当作全市场已验证。

## 2. 报表 `income / balance / cashflow`
聚宽 `STK_INCOME_STATEMENT / STK_BALANCE_SHEET / STK_CASHFLOW_STATEMENT` 原样，去掉了 `id`。关键列：

| 列 | 含义 |
|---|---|
| `pub_date` | **公告日** |
| `end_date` | 报告期末（`start_date` 是期初，仅利润表有） |
| `report_type` | 0 = 首次披露；1 = 后续期报里出现的上期对比数据，`pub_date` 是后一份报告的公告日。（从平安银行实例推断，聚宽未逐项说明） |
| `source` | 来源：定期报告 387,091、招募说明书 13,631、预披露公告 9,994、上市公告书 1,548（利润表计数）。**因子研究请取 `source='定期报告'` 且 `report_type=0`** |
| `source_id`、`report_date` | 来源编号、报告期 |

- 科目是**累计口径**：一季报 = Q1，半年报 = 前两季，三季报 = 前三季，年报 = 全年。
- 同一 (code, end_date, pub_date, report_type) 有 14~38 行重复，是同一天多个来源（如定期报告与更正公告）。
- `end_date` 最早到 2009 年（来自招股书中的历史报表），最新到 2026-06-30（三季报未到公告期）。

## 3. 季度财务摘要 `financials_pit`（推荐入口）
取 `report_type=0` 且 `source='定期报告'`，同一 (code, end_date) 有多个公告日时保留**最早**的一条（当时能看到的）。三张表的公告日 100% 一致。

| 类别 | 列 | 口径 |
|---|---|---|
| 键 | `code, end_date, pub_date` | 公告日、报告期 |
| 利润表（累计） | `total_operating_revenue, operating_revenue, operating_cost, operating_profit, total_profit, net_profit, np_parent_company_owners, rd_expenses, basic_eps` | 元 |
| 资产负债表（期末） | `total_assets, total_liability, total_owner_equities, equities_parent_company_owners, total_current_assets, total_current_liability, inventories, account_receivable, good_will, bs_pub_date` | 元 |
| 现金流量表（累计） | `net_operate_cash_flow, net_invest_cash_flow, net_finance_cash_flow, cf_pub_date` | 元 |
| 聚宽指标，**单季度** | `roe_q, roa_q, gross_profit_margin_q, net_profit_margin_q, inc_revenue_yoy_q, inc_net_profit_yoy_q, inc_np_parent_yoy_q, ocf_to_revenue_q, ind_pub_date` | %，单季度，不是累计也不是年化 |
| 我自己算的累计比率 | `gross_margin_cum, roe_cum, debt_ratio` | 毛利率 = (营业收入 − 营业成本) ÷ 营业收入；`roe_cum` = 归母净利润 ÷ 期末归母权益，**不年化**；资产负债率 |
| 我自己算的同比 | `revenue_yoy_cum, np_parent_yoy_cum` | 与去年同一报告期的累计值比；去年为非正数时置空 |
| 我自己算的 TTM | `np_parent_ttm, revenue_ttm, net_operate_cash_flow_ttm` | 本期累计 + 上一年报 − 去年同期累计；年报本身就是 TTM |

`_cum` / `_ttm` 字段只用当期和更早报告期的首次披露值，没有前视。核对：平安银行 2022 年归母净利润 455.2 亿、同比 +25.26%，2023 年 464.6 亿、+2.06%，与公开数据一致。
空值率（2017 年一季报起，167,903 行）：净利润、归母净利润、总资产、总负债、经营现金流 0%；`gross_margin_cum` 2.1%（银行等无营业成本）；三个 `_ttm` 字段 4.6%（缺上一年报或去年同期）；`np_parent_yoy_cum` 22.2%（去年同期为非正数时置空）。

## 4. 财务指标 `indicator`
聚宽 `get_fundamentals(query(indicator), statDate='YYYYqN')` 逐季拉取，报告期 2014q4 至 2026q2。列含 `eps, roe, roa, 毛利率, 各类费用率, 同比` 等 31 个指标，加 `report_quarter, end_date, pub_date`。
**重要：这是单季度口径**，不是累计。实测平安银行 2023q4 的 `eps=0.3514`，而全年 EPS 2.25、前三季 1.94，只有差值 ≈ 0.31 才接近；ROE 同理（Q4 为 1.45%，全年约 9.8%）。名称里含 `annual` 的列聚宽文档里是环比增长，我没有逐项核对。
`pub_date` 来自指标表自带的 `pubDate`；它的数值可能是后续修订版，**不保证是首次披露值**。要严格防前视，请用 `financials_pit` 里自己算的 `_cum` / `_ttm` 字段。

## 5. 业绩预告 `forecast`
`code, name, end_date`（预告对应报告期）, `pub_date`（预告公告日）, `report_type`（一季度/半年度/三季度/四季度预告）, `type`（预告类型：业绩大幅上升 22,810、业绩预亏 15,555、业绩预增 10,671、业绩大幅下降 9,674、预计扭亏 7,626 等）, `profit_min, profit_max`（预告净利润下限/上限）, `profit_last`（上年同期）, `profit_ratio_min, profit_ratio_max`（预告同比区间 %）, `content`（公告正文）。
**业绩快报聚宽没有**，所以本数据集缺快报。

## 6. 股东户数 `holder_num`
`code, end_date`（截止日）, `pub_date`（公告日）, `share_holders`（总户数）, `a_share_holders, b_share_holders, h_share_holders`。

## 7. 限售解禁
- `unlimit_events`：已发生的解禁，`pub_date` 公告日、`actual_unlimited_date` 实际解禁日、`actual_unlimited_number` 解禁股数、`actual_unlimited_ratio` 占总股本 %、`limited_reason`（股权激励、承诺限售等）、`shareholder_name`。
- `unlimit_schedule`：解禁日程，含 `expected_unlimited_date / number / ratio`（预计）与 `actual_*`（实际，未发生则为空）、`trade_condition`。**预计解禁日在 2026-10-01 之后的有 4,587 行，最晚到 2121-07-03**（远期占位日期，使用前请截断到合理期限，如未来 90 天）。

## 已知限制
- **聚宽没有业绩快报**，清单里的这一项缺失。
- 聚宽指标表是单季度口径且可能含修订值；估值表的公告日对齐只抽查了一只股票。
- 报表的 `report_type` 含义是我从一个样例推断的，不是聚宽文档的说明。
- 个股估值里的负数和极端值要自己处理；`pcf_ratio` 负数占比 46%，我没有查原因。
- 数据到 2026-09-30，三季报还没有公告，最新报告期是 2026-06-30。
