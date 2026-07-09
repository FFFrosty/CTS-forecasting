# CTS-forecasting

CTS 2026 算法大赛 — 港口拖轮 AIS 数据预测

## 赛题

基于拖轮 AIS 数据：
- **赛题A**：区域活跃拖轮数量预测（1小时窗口 × 3圈层）
- **赛题B**：圈层间拖轮迁移量预测（1小时窗口 × 6迁移方向）

## 项目结构

```
├── data/                   # 数据目录（不入 git）
│   ├── raw/                # 原始数据
│   └── processed/          # 清洗后数据
├── notebooks/              # EDA 分析
├── src/
│   ├── data/               # 数据加载与清洗
│   ├── features/           # 特征工程
│   ├── models/             # 预测模型
│   └── submission.py       # 提交文件生成
├── scripts/                # 执行入口
├── tests/                  # 单元测试
└── configs/                # 配置文件
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
