"""训练与预测 v4：A/B 任务独立的整数统计融合。"""
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_validation_count
from src.evaluation import (
    predict_group_mean,
    predict_recent_hour_mean,
)
from src.models.ensemble import predict_integer_blend
from src.submission import generate_submissions, validate_prediction_grid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "submission" / "v4"

A_DATA_GAP = (pd.Timestamp("2018-01-12"), pd.Timestamp("2018-01-18 23:00"))
B_DATA_GAP = (pd.Timestamp("2018-01-13"), pd.Timestamp("2018-01-18 23:00"))
B_TERMINAL_CENSORED = (
    pd.Timestamp("2018-01-24 23:00"),
    pd.Timestamp("2018-01-24 23:00"),
)

A_GLOBAL_WEIGHT = 0.3
A_RECENT_WEIGHT = 0.7
B_GLOBAL_WEIGHT = 0.7
B_RECENT_WEIGHT = 0.3

ZONES = [("核心区",), ("近港区",), ("外围区",)]
DIRECTIONS = [
    ("核心区", "近港区"), ("核心区", "外围区"),
    ("近港区", "核心区"), ("近港区", "外围区"),
    ("外围区", "核心区"), ("外围区", "近港区"),
]


def load_processed(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DIR / filename,
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )


def exclude_ranges(
    df: pd.DataFrame,
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    for start, end in ranges:
        keep &= ~df["time_window"].between(start, end)
    return df.loc[keep].copy()


def build_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    task_a = exclude_ranges(load_processed("task_a_train.csv"), [A_DATA_GAP])
    task_b = exclude_ranges(
        load_processed("task_b_train.csv"),
        [B_DATA_GAP, B_TERMINAL_CENSORED],
    )
    validation_counts = load_validation_count(
        RAW_DIR / "验证集_20180125-0131_每日拖轮数量.csv"
    )
    forecast_times = pd.date_range(
        validation_counts["date"].min(),
        validation_counts["date"].max() + pd.Timedelta(hours=23),
        freq="h",
    )

    global_mean = partial(predict_group_mean, round_predictions=False)
    recent_14 = partial(
        predict_recent_hour_mean, n_days=14, round_predictions=False
    )
    recent_10 = partial(
        predict_recent_hour_mean, n_days=10, round_predictions=False
    )
    task_a_predictions = predict_integer_blend(
        task_a,
        forecast_times,
        ["zone"],
        predictors=[global_mean, recent_14],
        weights=[A_GLOBAL_WEIGHT, A_RECENT_WEIGHT],
    )
    task_b_predictions = predict_integer_blend(
        task_b,
        forecast_times,
        ["source_zone", "target_zone"],
        predictors=[global_mean, recent_10],
        weights=[B_GLOBAL_WEIGHT, B_RECENT_WEIGHT],
    )
    validate_prediction_grid(task_a_predictions, forecast_times, ["zone"], ZONES)
    validate_prediction_grid(
        task_b_predictions,
        forecast_times,
        ["source_zone", "target_zone"],
        DIRECTIONS,
    )
    return task_a_predictions, task_b_predictions


def main() -> None:
    task_a_predictions, task_b_predictions = build_predictions()
    paths = generate_submissions(
        task_a_predictions,
        task_b_predictions,
        template_dir=RAW_DIR,
        output_dir=OUTPUT_DIR,
    )
    print(f"Task A: {len(task_a_predictions)} rows, total={task_a_predictions['predicted'].sum()}")
    print(f"Task B: {len(task_b_predictions)} rows, total={task_b_predictions['predicted'].sum()}")
    print(f"Submission A: {paths[0]}")
    print(f"Submission B: {paths[1]}")


if __name__ == "__main__":
    main()
