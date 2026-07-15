# CTS Forecasting

*CTS 2026 港口拖轮 AIS 预测项目，同时完成活跃拖轮数量预测与圈层迁移量预测。*

---

## 📌 项目概览

项目使用 2018 年 1 月 1 日至 24 日的拖轮 AIS 轨迹，预测 1 月 25 日至 31 日每小时的两个目标。港区以 `(117.79°E, 38.97°N)` 为中心，划分为核心区（0–3 km）、近港区（3–10 km）和外围区（10–30 km）。

| 任务 | 官方口径 | 提交规模 |
| --- | --- | ---: |
| A：活跃拖轮数量 | 同一船、同一小时、同一圈层内，`2 <= SOG <= 10` 的 AIS 记录不少于 3 条；一艘船可在多个圈层分别计数 | 168 小时 × 3 圈层 = 504 行 |
| B：圈层迁移量 | 每船每小时取 AIS 点数最多的代表圈层；并列时取最后出现时间更晚的圈层；相邻小时代表圈层不同则计一次迁移 | 168 小时 × 6 方向 = 1008 行 |

B 题不使用 A 题的 SOG 和最少记录数筛选。提交系统只接受非负整数预测，生成脚本会校验完整网格并输出整数列。

## 🚀 快速开始

项目约定使用本地 Conda 环境 `CTS2026`，需要 Python 3.10 或更高版本。

将以下四个官方文件放入 `data/raw/`：

- `训练集_20180101-0124_拖轮AIS.csv`
- `验证集_20180125-0131_每日拖轮数量.csv`
- `提交结果1_区域活跃拖轮数量.csv`
- `提交结果2_圈层间拖轮迁移量.csv`

```powershell
conda activate CTS2026
python -m pip install -e ".[dev]"
python scripts/preprocess.py
python scripts/train_v4.py
python -m pytest -q
```

生成的当前候选结果位于：

- `data/submission/v4/task_a_submission.csv`
- `data/submission/v4/task_b_submission.csv`

## 🏗️ 数据与建模流程

```mermaid
flowchart LR
    accTitle: CTS Forecasting Data Pipeline
    accDescr: Raw AIS data is cleaned into task-specific hourly labels, evaluated on calendar-aware folds, blended with separate A and B weights, and written as integer submissions

    raw_data[(📥 原始数据)] --> preprocess[⚙️ 清洗与标注]
    preprocess --> hourly_grid[(💾 小时完整网格)]
    hourly_grid --> calendar_backtest[🧪 日历回测]
    calendar_backtest --> weight_search[🔎 分题权重搜索]
    weight_search --> train_v4[⚙️ 生成 v4]
    train_v4 --> submissions([📤 整数提交文件])

    classDef data fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef result fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class raw_data,hourly_grid data
    class preprocess,calendar_backtest,weight_search,train_v4 process
    class submissions result
```

预处理会生成完整的零值小时网格，而不是只保留非零标签：A 题 1728 行，B 题 3456 行。训练质量控制按任务分开：

| 任务 | 排除的训练标签 | 原因 |
| --- | --- | --- |
| A | 1 月 12 日至 18 日 | 主数据源覆盖从 1 月 12 日起明显下降，最少 3 条记录的活跃口径会系统性漏计 |
| B | 1 月 13 日至 18 日 | 主数据源断档会影响代表圈层和迁移标签 |
| B | 1 月 24 日 23:00 | 原始数据止于当日 23:59，无法观察到下一小时，标签右删失 |

## 📊 当前模型与回测

v4 为 A/B 独立的整数统计融合：

| 任务 | 全历史分组均值 | 近期同小时均值 |
| --- | ---: | ---: |
| A | 30% | 最近 14 个有效日，70% |
| B | 70% | 最近 10 个有效日，30% |

统一回测使用 5 个可用的连续 3 日窗口，起点为 1 月 8、9、19、20、21 日；评价指标为 `SSE_A + 3 × SSE_B`。下表是每折平均值：

| 策略 | 加权 SSE | A SSE | B SSE | A MAE | B MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 任务独立整数融合 | 3153.20 | 2004.80 | 382.80 | 2.322 | 0.532 |
| v3 校准统计基线 | 3378.80 | 2120.00 | 419.60 | 2.344 | 0.560 |
| 最近 10 个有效日同小时均值 | 3433.60 | 2149.00 | 428.20 | 2.375 | 0.568 |
| v2 日总量与小时比例 | 3453.00 | 2141.40 | 437.20 | 2.345 | 0.569 |

这些数字只代表本地日历回测，不等于线上成绩。已知 v3 在线上 A、B 两题都比 v2 略差，因此当前不推荐继续提交 v3；v4 尚需用线上成绩验证。

运行完整比较与权重搜索：

```powershell
python scripts/evaluate.py
python scripts/search_ensemble.py
```

## 🗂️ 关键文件

| 路径 | 作用 |
| --- | --- |
| `configs/settings.yaml` | 港区中心、圈层半径、活跃阈值和日期范围 |
| `scripts/preprocess.py` | 清洗、A/B 标签构建、完整网格与时序特征 |
| `scripts/evaluate.py` | 统一的日历感知加权 SSE 回测 |
| `scripts/search_ensemble.py` | 分别搜索 A、B 的整数融合权重 |
| `scripts/train_v4.py` | 生成当前 v4 提交文件 |
| `src/features/zone.py` | A 多圈层计数与 B 代表圈层语义 |
| `src/evaluation.py` | 回测折、预测器和评价指标实现 |
| `src/models/` | 统计基线、校准模型与整数融合 |
| `src/submission.py` | 模板对齐、网格校验和 CSV 输出 |
| `tests/` | 任务语义、评价、校准和融合测试 |

更完整的数据结论与五步优化进度见 `docs/plan.md`，赛题原始说明见 `docs/request.md`。

## 🔧 常见问题

如果脚本提示缺少 `data/processed/*.csv`，先运行 `python scripts/preprocess.py`。如果从项目目录外无法 `import src`，重新执行 `python -m pip install -e ".[dev]"`。不要把 1 月 12 日至 18 日的低覆盖标签直接当作真实业务低谷。
