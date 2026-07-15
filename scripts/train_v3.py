"""训练与预测 v3：每日船舶数校准的整数统计基线。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_validation_count
from src.models.calibrated import predict_calibrated_hour_mean
from src.submission import generate_submissions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "submission" / "v3"

A_DATA_GAP = (pd.Timestamp("2018-01-12"), pd.Timestamp("2018-01-18 23:00"))
B_DATA_GAP = (pd.Timestamp("2018-01-13"), pd.Timestamp("2018-01-18 23:00"))
B_TERMINAL_CENSORED = (
    pd.Timestamp("2018-01-24 23:00"),
    pd.Timestamp("2018-01-24 23:00"),
)
N_DAYS = 10

ZONES = [("核心区",), ("近港区",), ("外围区",)]
DIRECTIONS = [
    ("核心区", "近港区"), ("核心区", "外围区"),
    ("近港区", "核心区"), ("近港区", "外围区"),
    ("外围区", "核心区"), ("外围区", "近港区"),
]


def load_processed(filename: str, time_col: str = "time_window") -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DIR / filename,
        encoding="utf-8-sig",
        parse_dates=[time_col],
    )


def exclude_ranges(
    df: pd.DataFrame,
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    for start, end in ranges:
        keep &= ~df["time_window"].between(start, end)
    return df.loc[keep].copy()


def load_all_daily_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_counts = load_processed("daily_vessel_counts.csv", time_col="date")
    validation_path = RAW_DIR / "验证集_20180125-0131_每日拖轮数量.csv"
    validation_counts = load_validation_count(validation_path)
    counts = pd.concat([train_counts, validation_counts], ignore_index=True)
    counts["date"] = counts["date"].dt.normalize()
    if counts["date"].duplicated().any():
        raise ValueError("daily vessel counts contain duplicate dates")
    return counts.sort_values("date"), validation_counts


def validate_prediction_grid(
    predictions: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    expected_groups: list[tuple],
) -> None:
    keys = ["time_window"] + group_cols
    observed_groups = set(
        predictions[group_cols].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_groups != set(expected_groups):
        raise ValueError("prediction groups do not match the official template")
    if predictions.duplicated(keys).any():
        raise ValueError("prediction keys must be unique")
    if len(predictions) != len(forecast_times) * len(expected_groups):
        raise ValueError("prediction grid is incomplete")
    if set(predictions["time_window"]) != set(forecast_times):
        raise ValueError("prediction timestamps do not match the validation period")
    if not pd.api.types.is_integer_dtype(predictions["predicted"]):
        raise ValueError("submission predictions must be integers")
    if (predictions["predicted"] < 0).any():
        raise ValueError("submission predictions must be non-negative")


def build_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    task_a = load_processed("task_a_train.csv")
    task_b = load_processed("task_b_train.csv")
    task_a = exclude_ranges(task_a, [A_DATA_GAP])
    task_b = exclude_ranges(task_b, [B_DATA_GAP, B_TERMINAL_CENSORED])

    daily_counts, validation_counts = load_all_daily_counts()
    forecast_times = pd.date_range(
        validation_counts["date"].min(),
        validation_counts["date"].max() + pd.Timedelta(hours=23),
        freq="h",
    )

    task_a_predictions = predict_calibrated_hour_mean(
        task_a,
        forecast_times,
        ["zone"],
        daily_vessel_counts=daily_counts,
        n_days=N_DAYS,
    )
    task_b_predictions = predict_calibrated_hour_mean(
        task_b,
        forecast_times,
        ["source_zone", "target_zone"],
        daily_vessel_counts=daily_counts,
        n_days=N_DAYS,
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
