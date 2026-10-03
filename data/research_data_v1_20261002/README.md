# research_data_v1_20261002 — 研究用基础数据（分红 / 股票状态 / 行业分类）

生成于 2026-10-02。Parquet 与 CSV 两份内容相同（UTF-8，表头，逗号分隔，日期为 YYYY-MM-DD）。`raw/` 里是原始来源文件和生成脚本，可复现。

| 文件 | 行数 | 覆盖 | 一句话 |
|---|---|---|---|
| `dividends` | 82363 | 2018-01-03 ~ 2026-10-15（除权日；公告到 2026-10-01）| 每个分红方案一行：公告日、阶段、登记/除权/派息日、每股现金、送转比例 |
| `listing` | 5677 | 1990-12-01 ~ 2026-09-30 | 每只证券一行：上市日、退市日（含已退市）|
| `st_events` | 11948 | 1990-12-10 ~ 2026-09-30 | 聚宽状态变动原始事件（ST/*ST/摘帽/退市整理/终止上市…），带公告日和生效日 |
| `status_intervals` | 9911 | 同上 | 由事件推出的状态区间（**ST 覆盖不全，见下**）|
| `st_daily_intervals` | 1180 | 2011-10-10 ~ 2026-09-30 | **由逐日 is_st 标记生成的 ST 区间，判断某天是否 ST 以这张为准** |
| `suspensions` | 22555 | 2011-10-10 ~ 2026-09-30 | 全天停牌的连续区间与复牌日 |
| `industry_intervals` | 49717 | 2017-01-26 ~ 2026-09-30（月末采样）| 6 套行业分类体系的生效区间 |

证券代码统一为聚宽格式：`000001.XSHE`（深）/ `600000.XSHG`（沪）。

## 各表数据截止日期（取数日 2026-10-02，最近交易日 2026-09-30）

| 表 | 截止 |
|---|---|
| dividends | 公告到 2026-10-01，除权日到 2026-10-15 |
| listing / st_events / status_intervals | 2026-09-30 |
| st_daily_intervals / suspensions | 2026-09-30 |
| industry_intervals | 最后采样日 2026-09-30 |

## 先看这几条限制（会影响回测结论）

1. **ST 状态用 `st_daily_intervals`**。聚宽事件表（`st_events`）推出的 `status_intervals` 只覆盖逐日 `is_st` 标记的 96.4%，漏了 160 只证券共 11,526 个证券日（例：002260 在「戴帽披星」之后进入暂停上市，事件表里没有中间的 ST 记录）；没有误报。事件表的独有价值是**公告日**和**生效日**。
2. **行业生效日精度是月**。聚宽只能按日期取当时的分类，没有生效日字段；我用月末采样，`effective_from` 是「首次在采样日观察到」，真实生效日在它之前、上一个采样日之后，**最多晚一个月**（例：申万 2021 版调整，银行在 2021-12-31 首次观察到新分类）。需要精确到日要对变化区间再做日度二分，没做。
3. **分红金额口径**：`cash_per_10_shares_pretax_cny` 是**每 10 股、税前、元**（和交易所公告口径一致）；`cash_per_share_pretax_cny` = 前者 ÷ 10，是**每股**。算股息率用后者。送股、转增也是**每 10 股**。
4. **信息可得性**：回测里用「某日已知」的信息时，分红用 `implementation_pub_date`（实施公告日）而不是除权日；状态用 `announce_date` 而不是 `effective_date`；行业区间有最多一个月的滞后，用 `effective_from`。
5. **停牌**只含**全天**停牌，不含盘中临停；`suspend_start_date` 是第一个停牌的交易日，不是公告日。
6. 已退市证券都包含在内（`listing` 里 374 只有退市日；行业区间 333 只已退市证券有记录；停牌、ST 数据含已退市）。

## 字段说明

### dividends（分红明细）
来源：聚宽 `finance.STK_XR_XD` 整表（2026-10-02 取，`raw/stk_xr_xd.csv.gz`）。本地 `dividend_allA_2013_20260906_v1/actions.parquet` 是它在 2026-09-06 的导出，字段完全相同；本地 122,662 个方案在聚宽里全部存在，聚宽另有 09-06 之后的新公告和阶段推进。一行 = 一个分红方案（证券 + 报告期 + 分红类型），`status` 是它**当前所处阶段**。入选条件：除权日 / 实施公告日 / 股东大会预案日 / 董事会预案日 任一落在 2018-01-01 之后。**2025、2026 年也包含**（按除权年的实施方案数：2018 年 2,901、2019 年 2,712、2020 年 2,832、2021 年 3,179、2022 年 3,408、2023 年 3,565、2024 年 4,544、2025 年 4,661、2026 年 4,153）。**2026 年不是完整年度**：数据取于 2026-10-02，最新公告日是 2026-10-01，最晚除权日 2026-10-15；之后公告的方案不在其中。

| 字段 | 类型 | 含义 |
|---|---|---|
| code | 文本 | 证券代码 |
| company_name | 文本 | 公司名称 |
| report_period_end | 文本 | 报告期截止日（如 2024-12-31 = 2024 年报，2024-06-30 = 中报）|
| bonus_type | 文本 | 年度分红 / 中期分红 / 季度分红 / 特别分红 / 重整转增 / 股改分红 / 承诺补偿 |
| status | 文本 | 实施阶段：董事会预案 / 股东大会预案 / 实施方案（已实施或已公告实施）/ 取消分红 / 延迟实施 / 终止 |
| board_plan_pub_date | 日期 | 董事会预案公告日 |
| shareholders_plan_pub_date | 日期 | 股东大会预案/决议公告日（可能为空）|
| implementation_pub_date | 日期 | 实施公告日（回测里分红「已知」的日期）|
| record_date | 日期 | A 股股权登记日 |
| ex_date | 日期 | A 股除权除息日（预案阶段为空；实施方案中有 103 行为空，多为刚公告尚未定除权日的）|
| pay_date | 日期 | 派息日（缺失较多）|
| cash_per_10_shares_pretax_cny | 数 | **每 10 股**派现，税前，元 |
| cash_per_share_pretax_cny | 数 | **每股**派现，税前，元 = 上一列 ÷ 10 |
| bonus_shares_per_10 | 数 | **每 10 股**送股数 |
| transfer_shares_per_10 | 数 | **每 10 股**转增数 |
| shares_added_per_10 | 数 | 送股 + 转增（空值按 0 计）|
| share_base_10k_shares | 数 | 分配基数，**万股** |
| cash_total_10k_cny | 数 | 现金分红总额，**万元**（≈ 每 10 股派现 ÷ 10 × 分配基数）|

### listing（上市 / 退市）
来源：聚宽研究环境 `finance.STK_LIST`（`raw/stk_list.csv`）。含 A 股与 B 股、已退市证券。

| 字段 | 含义 |
|---|---|
| code, security_name, short_name | 代码、名称、拼音简称 |
| share_class | A / B |
| exchange | XSHE / XSHG |
| list_date | 上市日 |
| delist_date | 退市（终止上市）日，仍在市为空 |
| is_delisted | 是否已退市 |
| latest_listing_status | 当前状态（正常上市 / ST / *ST / 终止上市 / 进入退市整理期）|
| company_name | 公司全称 |

### st_events（状态事件，原始）与 status_intervals（状态区间）
来源：聚宽 `finance.STK_STATUS_CHANGE`（`raw/stk_status_change.csv`）。

| 字段 | 含义 |
|---|---|
| announce_date | 公告日 |
| effective_date | **生效日** |
| change_type | 新股上市 / 戴帽（ST）/ 戴帽披星（*ST）/ 披星 / 摘星 / 摘帽 / 摘星摘帽 / 暂停上市 / 恢复上市 / 退市整理 / 终止上市 / 重新上市 / 转板上市 … |
| change_reason | 原因文本 |
| status_after | 事件后的状态：正常上市 / ST / *ST / 暂停上市 / 进入退市整理期 / 终止上市 |

`status_intervals`：`status` 从 `start_date` 起持续到 `end_date_exclusive`（不含；空 = 至今）。已剔除上市前的「拟上市 / 预披露 / 暂缓发行 / 未过会 / 发行失败」。**ST 的完整性见限制 1。**

### st_daily_intervals（ST 区间，判断用这张）
来源：`prices.parquet` 的 `is_st`（到 2026-09-04）+ 聚宽 `get_extras('is_st')`（2026-09-01 ~ 09-30，`raw/is_st_202609.csv`）；两者在 09-01~09-04 的 20,859 个证券日上完全一致。

| 字段 | 含义 |
|---|---|
| st_start_date | 连续 ST 的第一个交易日 |
| last_st_date | 最后一个 ST 交易日 |
| st_end_date_exclusive | 第一个非 ST 的交易日；**空 = 到 2026-09-30 仍是 ST** |
| st_trading_days | 区间内交易日数 |

### suspensions（停复牌）
来源：`prices.parquet` 的 `paused`（到 2026-09-04）+ `stock_price` 收盘价为空的行（09-05 起）；两者在 860 万个重叠证券日上完全一致。

| 字段 | 含义 |
|---|---|
| suspend_start_date | 第一个全天停牌的交易日 |
| last_suspended_date | 最后一个停牌交易日 |
| resume_date | 复牌后第一个交易日；空 = 数据末端仍停牌或已退市 |
| suspended_trading_days | 区间内的停牌交易日数 |
| still_suspended_at_data_end | resume_date 为空的标记 |

### industry_intervals（行业分类）
来源：聚宽 `get_industry(…, date)`，2017-01 ~ 2026-09 每月最后一个交易日采样（`raw/industry_samples_*.csv.gz`），把同一只票同一体系里连续相同的行业折叠成区间。聚宽的交易日历包含未来日期，2026-09-30 之后的采样已剔除。

| 字段 | 含义 |
|---|---|
| classification_system | `sw_l1/sw_l2/sw_l3` 申万一二三级；`jq_l1/jq_l2` 聚宽一二级；`zjw` 证监会行业 |
| industry_code, industry_name | 行业代码与名称（体系改版时同名行业的代码可能变，**统计请用名称，并留意改版日期**，如申万 2021-12）|
| effective_from | 首次在采样日观察到的日期（真实生效日最多早一个月）|
| last_observed | 最后一次观察到的采样日 |
| effective_to_exclusive | 下一个不同行业的首次观察日；空 = 到 2026-09-30 仍是它 |

本目录**不含同花顺行业**；同花顺逐月快照在 `tick_parquet/ths_industry.parquet`（2018-01 起，有 industry_code 会换的问题，见项目记忆）。

## 已做的交叉验证
- 停复牌两来源：860 万重叠证券日完全一致（28,482 个停牌日两边都判停牌，0 个分歧）。
- 逐日 ST：聚宽 9 月标记与本地 `is_st` 重叠 20,859 个证券日完全一致。
- 分红：实施方案的除权日与 `stock_price` 的 `pre_close` 跳变对照，召回 99.0%，理论除权价与 `pre_close` 相差 ≤0.011 元的占 97.1%。
- 状态事件区间 vs 逐日 `is_st`：覆盖 96.4%，无误报（见限制 1）。

## 复现
`raw/build_research_data_v1.py <表名>`（dividends / listing / st_events / suspensions / st_daily_intervals / industry）。聚宽原始数据由 `tauri-kline/scripts/jq_research.py` 取得。
