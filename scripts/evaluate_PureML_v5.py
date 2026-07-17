"""按各版本真实训练规则，在多个预测长度上比较 v2/v4/PureML。"""
import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    evaluate_backtest,
    exclude_time_ranges,
    predict_daily_profile,
    predict_group_mean,
    predict_recent_hour_mean,
)
from src.models.ensemble import predict_integer_blend
from src.models.tree import MODEL_NAMES, predict_pure_ml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
A_DATA_GAP = (pd.Timestamp("2018-01-12"), pd.Timestamp("2018-01-18 23:00"))
B_DATA_GAP = (pd.Timestamp("2018-01-13"), pd.Timestamp("2018-01-18 23:00"))
B_TERMINAL_CENSORED = (
    pd.Timestamp("2018-01-24 23:00"),
    pd.Timestamp("2018-01-24 23:00"),
)
EXCLUDED_TEST_RANGES = [A_DATA_GAP, B_TERMINAL_CENSORED]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[*MODEL_NAMES, "all"],
        default="lightgbm",
        help="默认只评估 LightGBM；RF 需显式指定。",
    )
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


def _is_task_a(group_cols: list[str]) -> bool:
    if group_cols == ["zone"]:
        return True
    if group_cols == ["source_zone", "target_zone"]:
        return False
    raise ValueError(f"unknown task group columns: {group_cols}")


def predict_v2_exact(
    train_df: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
) -> pd.DataFrame:
    """复现 v2：A/B 均仅排除 1 月 13–18 日。"""
    filtered = exclude_time_ranges(train_df, [B_DATA_GAP])
    return predict_daily_profile(
        filtered,
        forecast_times,
        group_cols,
        n_days=10,
        round_predictions=True,
    )


def predict_v4_exact(
    train_df: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
) -> pd.DataFrame:
    """复现 v4 的分题过滤范围与整数融合权重。"""
    if _is_task_a(group_cols):
        filtered = exclude_time_ranges(train_df, [A_DATA_GAP])
        recent = partial(
            predict_recent_hour_mean,
            n_days=14,
            round_predictions=False,
        )
        weights = [0.3, 0.7]
    else:
        filtered = exclude_time_ranges(
            train_df,
            [B_DATA_GAP, B_TERMINAL_CENSORED],
        )
        recent = partial(
            predict_recent_hour_mean,
            n_days=10,
            round_predictions=False,
        )
        weights = [0.7, 0.3]

    global_mean = partial(predict_group_mean, round_predictions=False)
    return predict_integer_blend(
        filtered,
        forecast_times,
        group_cols,
        predictors=[global_mean, recent],
        weights=weights,
    )


def predict_pure_ml_exact(
    train_df: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    model_name: str,
    random_state: int,
) -> pd.DataFrame:
    """复现 PureML v5 的分题过滤范围后直接训练树模型。"""
    ranges = [A_DATA_GAP] if _is_task_a(group_cols) else [
        B_DATA_GAP,
        B_TERMINAL_CENSORED,
    ]
    filtered = exclude_time_ranges(train_df, ranges)
    return predict_pure_ml(
        filtered,
        forecast_times,
        group_cols,
        model_name=model_name,
        random_state=random_state,
    )


def build_strategies(
    model_selection: str,
    random_state: int,
) -> list[tuple[str, object]]:
    strategies: list[tuple[str, object]] = [
        ("v2", predict_v2_exact),
        ("v4", predict_v4_exact),
    ]
    model_names = MODEL_NAMES if model_selection == "all" else (model_selection,)
    for model_name in model_names:
        strategies.append((
            model_name,
            partial(
                predict_pure_ml_exact,
                model_name=model_name,
                random_state=random_state,
            ),
        ))
    return strategies


def main() -> None:
    args = parse_args()
    horizons = sorted(set(args.horizons))
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")

    task_a = load_task("task_a_train.csv")
    task_b = load_task("task_b_train.csv")
    strategies = build_strategies(args.model, args.random_state)
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
    print("\n多预测长度日历回测（每个策略使用其真实训练过滤规则）")
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
