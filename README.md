# feed

公开市场数据，版本日期 2026-10-02。

- `data/stock_price_v1/`：2019—2026 年度价格 Parquet，共 8 个文件。每个文件保留 24 列，已移除 `id`、`create_time`、`update_time`、`create_by`、`update_by`。
- `data/research_data_v1_20261002/`：7 个补充 Parquet、`dividends_source.csv.gz` 和原有 3 份说明/核验文件。目录内使用原始短文件名；使用说明中提及的 `research_data_v1_20261002_` 前缀对应此前打包名称。
- `CHECKSUMS.json`：19 个文件的字节大小及 SHA-256，总大小 403,144,885 字节。

使用前请阅读补充目录中的 `README.md` 和 `RESEARCH_USE_NOTES.txt`，注意来源、日期覆盖、历史信息可得性及修订版本限制。原 README 描述源目录；本仓库仅包含上述已公开上传的文件。

本次发布只传输已核验文件，未运行因子研究、选股、回测或训练，未读取 2025—2026 年价格值。
