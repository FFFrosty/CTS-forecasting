"""统一回测入口：在连续日历窗口上按官方加权 SSE 比较策略。"""
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    evaluate_backtest,
    predict_daily_profile,
    predict_group_mean,
    predict_hour_dow_mean,
    predict_recent_hour_mean,
)
from src.models.calibrated import predict_calibrated_hour_mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# A题从 1/12 起已受主数据源下降影响；B题从完全断档的 1/13 起排除。
A_DATA_GAP = (pd.Timestamp("2018-01-12"), pd.Timestamp("2018-01-18 23:00"))
B_DATA_GAP = (pd.Timestamp("2018-01-13"), pd.Timestamp("2018-01-18 23:00"))
# 原始训练集止于 1/24 23:59，无法计算 1/24 23:00 -> 1/25 00:00 的 B 题迁移。
B_TERMINAL_CENSORED = (
    pd.Timestamp("2018-01-24 23:00"),
    pd.Timestamp("2018-01-24 23:00"),
)

# A/B 的训练质量策略分开声明，后续可独立比较 B 题是否保留断层期。
A_EXCLUDED_TRAIN_RANGES = [A_DATA_GAP]
B_EXCLUDED_TRAIN_RANGES = [B_DATA_GAP, B_TERMINAL_CENSORED]
EXCLUDED_TEST_RANGES = [A_DATA_GAP, B_TERMINAL_CENSORED]

MIN_TRAIN_DAYS = 7
HORIZON_DAYS = 3


def load_task(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DIR / filename,
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )


def load_daily_counts() -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DIR / "daily_vessel_counts.csv",
        encoding="utf-8-sig",
        parse_dates=["date"],
    )


def main() -> None:
    task_a = load_task("task_a_train.csv")
    task_b = load_task("task_b_train.csv")
    daily_vessel_counts = load_daily_counts()

    strategies = [
        (
            "全历史分组均值（整数）",
            partial(predict_group_mean, round_predictions=True),
        ),
        (
            "同星期同时刻均值（取整）",
            partial(predict_hour_dow_mean, round_predictions=True),
        ),
        (
            "最近10个有效日同小时均值（整数）",
            partial(predict_recent_hour_mean, n_days=10, round_predictions=True),
        ),
        (
            "最近14个有效日同小时均值（整数）",
            partial(predict_recent_hour_mean, n_days=14, round_predictions=True),
        ),
        (
            "v2日总量+小时比例（取整）",
            partial(predict_daily_profile, n_days=10, round_predictions=True),
        ),
        (
            "每日船舶数校准同小时均值（整数）",
            partial(
                predict_calibrated_hour_mean,
                daily_vessel_counts=daily_vessel_counts,
                n_days=10,
            ),
        ),
    ]

    print("=" * 88)
    print("A/B 统一日历回测")
    print(f"最少训练期: {MIN_TRAIN_DAYS} 天；预测期: {HORIZON_DAYS} 天")
    print("指标: weighted_sse = SSE_A + 3 * SSE_B")
    print("所有策略输出非负整数；断层期不作为验证目标。")
    print("A题从1/12起排除低覆盖数据；B题最后一个训练小时视为右删失标签。")
    print("=" * 88)

    summaries = []
    fold_starts = None
    for name, predictor in strategies:
        result = evaluate_backtest(
            task_a=task_a,
            task_b=task_b,
            predictor=predictor,
            min_train_days=MIN_TRAIN_DAYS,
            horizon_days=HORIZON_DAYS,
            excluded_test_ranges=EXCLUDED_TEST_RANGES,
            a_excluded_train_ranges=A_EXCLUDED_TRAIN_RANGES,
            b_excluded_train_ranges=B_EXCLUDED_TRAIN_RANGES,
        )
        if result.empty:
            raise RuntimeError("没有可用回测折，请检查日期范围与排除区间")
        if fold_starts is None:
            fold_starts = result["forecast_start"].dt.strftime("%m-%d").tolist()

        summaries.append({
            "strategy": name,
            "folds": len(result),
            "weighted_sse": result["weighted_sse"].mean(),
            "weighted_mse": result["weighted_mse"].mean(),
            "a_sse": result["a_sse"].mean(),
            "b_sse": result["b_sse"].mean(),
            "a_mae": result["a_mae"].mean(),
            "b_mae": result["b_mae"].mean(),
        })

    summary = pd.DataFrame(summaries).sort_values("weighted_sse")
    print(f"回测起点: {', '.join(fold_starts)}")
    print("以下 SSE 均为各折平均值，所有策略使用完全相同的目标窗口。\n")
    print(summary.to_string(
        index=False,
        formatters={
            "weighted_sse": "{:.2f}".format,
            "weighted_mse": "{:.4f}".format,
            "a_sse": "{:.2f}".format,
            "b_sse": "{:.2f}".format,
            "a_mae": "{:.3f}".format,
            "b_mae": "{:.3f}".format,
        },
    ))


if __name__ == "__main__":
    main()
