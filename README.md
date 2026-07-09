# CTS-forecasting

CTS 2026 算法大赛 — 港口拖轮 AIS 数据预测

## 赛题

基于拖轮 AIS 数据：
- **赛题A**：区域活跃拖轮数量预测（1小时窗口 × 3圈层）
- **赛题B**：圈层间拖轮迁移量预测（1小时窗口 × 6迁移方向）

## 项目结构

```
CTS-forecasting/
├── .gitignore
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── settings.yaml              # 中心坐标、圈层半径、活跃阈值
├── data/
│   └── .gitkeep                    # raw/ 和 processed/ 不入git
├── notebooks/
│   └── 01_eda_overview.py         # 数据分析同学入口
├── src/
│   ├── data/
│   │   ├── loader.py              # load_training_data(), load_submission_template()
│   │   └── cleaner.py             # filter_tug_vessels(), clean_sentinels()
│   ├── features/
│   │   ├── spatial.py             # haversine_distance(), classify_zone()
│   │   ├── zone.py                # 活跃状态标注、赛题A/B样本构建
│   │   └── temporal.py            # 滞后特征、滚动统计
│   ├── models/
│   │   ├── baseline.py            # 历史均值、滚动均值外推
│   │   └── autoregressive.py      # ARIMA/SARIMA 预测
│   └── submission.py              # 填模板、输出提交 CSV
├── scripts/
│   ├── preprocess.py              # 清洗 → 特征 → 输出 processed/
│   └── train.py                   # 训练 → 预测 → 生成提交文件
└── tests/
    └── test_cleaner.py            # cleaner 单元测试
```

## 环境配置

```bash
pip install -r requirements.txt
pip install -e .
```

## 分工

- 数据分析：`notebooks/`
- 数据预处理 + 特征工程：`src/data/` `src/features/`
- 自回归建模：`src/models/autoregressive.py`
