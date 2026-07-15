"""在多个预测长度上分别搜索 A/B 两题的整数融合权重。"""
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    evaluate_backtest,
    predict_group_mean,
    predict_recent_hour_mean,
)
from src.models.ensemble import predict_integer_blend


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
A_DATA_GAP = (pd.Timestamp("2018-01-12"), pd.Timestamp("2018-01-18 23:00"))
B_DATA_GAP = (pd.Timestamp("2018-01-13"), pd.Timestamp("2018-01-18 23:00"))
B_TERMINAL_CENSORED = (
    pd.Timestamp("2018-01-24 23:00"),
    pd.Timestamp("2018-01-24 23:00"),
)
HORIZONS = [1, 2, 3]
WEIGHTS = np.arange(0.0, 1.01, 0.1)


def load_task(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DIR / filename,
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )


def make_blend(first, second, first_weight: float):
    return partial(
        predict_integer_blend,
        predictors=[first, second],
        weights=[first_weight, 1.0 - first_weight],
    )


def search_weights(
    task_a: pd.DataFrame,
    task_b: pd.DataFrame,
    first,
    second,
    score_col: str,
) -> pd.DataFrame:
    rows = []
    for weight in WEIGHTS:
        horizon_mse = []
        predictor = make_blend(first, second, float(weight))
        for horizon in HORIZONS:
            result = evaluate_backtest(
                task_a,
                task_b,
                predictor=predictor,
                min_train_days=7,
                horizon_days=horizon,
                excluded_test_ranges=[A_DATA_GAP, B_TERMINAL_CENSORED],
                a_excluded_train_ranges=[A_DATA_GAP],
                b_excluded_train_ranges=[B_DATA_GAP, B_TERMINAL_CENSORED],
            )
            task_name = score_col.removesuffix("_sse")
            horizon_mse.append(
                (result[score_col] / result[f"{task_name}_n"]).mean()
            )
        rows.append({
            "first_weight": round(float(weight), 1),
            "second_weight": round(1.0 - float(weight), 1),
            "mean_mse_h1_h3": float(np.mean(horizon_mse)),
        })
    return pd.DataFrame(rows).sort_values("mean_mse_h1_h3")


def main() -> None:
    task_a = load_task("task_a_train.csv")
    task_b = load_task("task_b_train.csv")
    global_mean = partial(predict_group_mean, round_predictions=False)
    recent_14 = partial(
        predict_recent_hour_mean, n_days=14, round_predictions=False
    )
    recent_10 = partial(
        predict_recent_hour_mean, n_days=10, round_predictions=False
    )

    task_a_search = search_weights(
        task_a, task_b, global_mean, recent_14, score_col="a_sse"
    )
    task_b_search = search_weights(
        task_a, task_b, global_mean, recent_10, score_col="b_sse"
    )
    print("A题：全历史圈层均值 + 最近14日同小时均值")
    print(task_a_search.head(5).to_string(index=False))
    print("\nB题：全历史方向均值 + 最近10日同小时均值")
    print(task_b_search.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
