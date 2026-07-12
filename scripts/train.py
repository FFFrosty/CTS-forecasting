"""训练与预测入口脚本。

1. 加载预处理后的样本
2. 训练自回归模型
3. 生成提交文件
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline import historical_mean_predict
from src.models.autoregressive import forecast_all_zones_arima
from src.submission import generate_submissions


def main():
    processed_dir = Path("data/processed")
    template_dir = Path("data/raw")
    output_dir = Path("data/submission")

    # 加载预处理样本
    task_a_train = pd.read_csv(
        processed_dir / "task_a_train.csv",
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )
    task_b_train = pd.read_csv(
        processed_dir / "task_b_train.csv",
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )

    # 排除数据源断层期（1/13-1/18），避免主源缺失导致的标签虚低污染均值
    # TODO 但是这样不好
    anom_start = pd.Timestamp("2018-01-13")
    anom_end = pd.Timestamp("2018-01-18 23:00")
    task_a_train = task_a_train[
        (task_a_train["time_window"] < anom_start)
        | (task_a_train["time_window"] > anom_end)
    ].copy()
    task_b_train = task_b_train[
        (task_b_train["time_window"] < anom_start)
        | (task_b_train["time_window"] > anom_end)
    ].copy()
    print(f"  Train samples after excluding data gap (1/13-1/18):")
    print(f"    Task A: {len(task_a_train)} rows")
    print(f"    Task B: {len(task_b_train)} rows")

    # ---- 赛题A：区域活跃拖轮数量 ----
    print("Task A: forecasting zone vessel counts...")
    # 基线：历史均值
    a_baseline = historical_mean_predict(
        task_a_train, forecast_horizon=168, group_cols=["zone"],
    )

    # 自回归模型（可选）
    # a_arima = forecast_all_zones_arima(
    #     task_a_train, forecast_horizon=168, group_cols=["zone"],
    # )

    a_predictions = a_baseline

    # ---- 赛题B：圈层间拖轮迁移量 ----
    print("Task B: forecasting zone migrations...")
    b_baseline = historical_mean_predict(
        task_b_train, forecast_horizon=168,
        group_cols=["source_zone", "target_zone"],
    )

    # 自回归模型（可选）
    # b_arima = forecast_all_zones_arima(
    #     task_b_train, forecast_horizon=168,
    #     group_cols=["source_zone", "target_zone"],
    # )

    b_predictions = b_baseline

    # ---- 生成提交文件 ----
    print("Generating submission files...")
    paths = generate_submissions(
        a_predictions, b_predictions,
        template_dir=template_dir,
        output_dir=output_dir,
    )
    print(f"  Task A submission: {paths[0]}")
    print(f"  Task B submission: {paths[1]}")
    print("Done.")


if __name__ == "__main__":
    main()
