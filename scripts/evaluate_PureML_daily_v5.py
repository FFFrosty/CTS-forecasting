"""按多个预测长度回测每日批量 PureML 的两个对照版本。"""
import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_PureML_v5 import (
    A_DATA_GAP,
    B_DATA_GAP,
    B_TERMINAL_CENSORED,
    EXCLUDED_TEST_RANGES,
    predict_v2_exact,
    predict_v4_exact,
)
from scripts.train_PureML_daily_v5 import mask_daily_counts
from src.evaluation import evaluate_backtest, exclude_time_ranges
from src.models.tree import predict_pure_ml_daily


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5],
        help="要评估的连续预测天数，默认 1/2/3/5 天。",
    )
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args()


def load_task(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DIR / filename,
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )


def predict_daily_exact(
    train_df: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    daily_counts: pd.DataFrame,
    include_daily_count: bool,
    random_state: int,
) -> pd.DataFrame:
    is_task_a = group_cols == ["zone"]
    label_ranges = [A_DATA_GAP] if is_task_a else [
        B_DATA_GAP,
        B_TERMINAL_CENSORED,
    ]
    filtered = exclude_time_ranges(train_df, label_ranges)
    masked_counts = mask_daily_counts(
        daily_counts,
        [A_DATA_GAP] if is_task_a else [B_DATA_GAP],
    )
    return predict_pure_ml_daily(
        filtered,
        forecast_times,
        group_cols,
        random_state=random_state,
        daily_vessel_counts=masked_counts if include_daily_count else None,
    )


def main() -> None:
    args = parse_args()
    horizons = sorted(set(args.horizons))
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")

    task_a = load_task("task_a_train.csv")
    task_b = load_task("task_b_train.csv")
    daily_counts = pd.read_csv(
        PROCESSED_DIR / "daily_vessel_counts.csv",
        encoding="utf-8-sig",
        parse_dates=["date"],
    )
    strategies = [
        ("v2", predict_v2_exact),
        ("v4", predict_v4_exact),
        (
            "daily_no_count",
            partial(
                predict_daily_exact,
                daily_counts=daily_counts,
                include_daily_count=False,
                random_state=args.random_state,
            ),
        ),
        (
            "daily_with_count",
            partial(
                predict_daily_exact,
                daily_counts=daily_counts,
                include_daily_count=True,
                random_state=args.random_state,
            ),
        ),
    ]
    summaries = []

    for horizon in horizons:
        print(f"Evaluating {horizon}-day horizon...")
        for strategy_name, predictor in strategies:
            result = evaluate_backtest(
                task_a,
                task_b,
                predictor=predictor,
                min_train_days=7,
                horizon_days=horizon,
                excluded_test_ranges=EXCLUDED_TEST_RANGES,
            )
            if result.empty:
                print(f"  {strategy_name}: no valid folds")
                continue
            summaries.append({
                "horizon_days": horizon,
                "model": strategy_name,
                "folds": len(result),
                "weighted_sse": result["weighted_sse"].mean(),
                "weighted_mse": result["weighted_mse"].mean(),
                "a_mse": (result["a_sse"] / result["a_n"]).mean(),
                "b_mse": (result["b_sse"] / result["b_n"]).mean(),
                "a_mae": result["a_mae"].mean(),
                "b_mae": result["b_mae"].mean(),
                "fold_starts": ",".join(
                    result["forecast_start"].dt.strftime("%m-%d")
                ),
            })

    if not summaries:
        raise RuntimeError("没有可用回测折，请检查预测长度和排除区间")

    summary = pd.DataFrame(summaries).sort_values(
        ["horizon_days", "weighted_mse"]
    )
    print("\n每日批量模型多预测长度日历回测")
    print(summary.to_string(
        index=False,
        formatters={
            "weighted_sse": "{:.2f}".format,
            "weighted_mse": "{:.4f}".format,
            "a_mse": "{:.4f}".format,
            "b_mse": "{:.4f}".format,
            "a_mae": "{:.3f}".format,
            "b_mae": "{:.3f}".format,
        },
    ))

    ranking = (
        summary.groupby("model", as_index=False)
        .agg(
            horizons=("horizon_days", "count"),
            mean_weighted_mse=("weighted_mse", "mean"),
            worst_weighted_mse=("weighted_mse", "max"),
        )
        .sort_values("mean_weighted_mse")
    )
    print("\n跨预测长度稳定性汇总（各长度等权，仅用于方法筛选）")
    print(ranking.to_string(
        index=False,
        formatters={
            "mean_weighted_mse": "{:.4f}".format,
            "worst_weighted_mse": "{:.4f}".format,
        },
    ))


if __name__ == "__main__":
    main()
