"""训练按天递归的 PureML v5，并生成有/无每日拖轮总数的两组提交。"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_validation_count
from src.evaluation import exclude_time_ranges
from src.models.tree import predict_pure_ml_daily
from src.submission import generate_submissions, validate_prediction_grid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "submission" / "PureML_v5" / "daily_batch"

A_DATA_GAP = (pd.Timestamp("2018-01-12"), pd.Timestamp("2018-01-18 23:00"))
B_DATA_GAP = (pd.Timestamp("2018-01-13"), pd.Timestamp("2018-01-18 23:00"))
B_TERMINAL_CENSORED = (
    pd.Timestamp("2018-01-24 23:00"),
    pd.Timestamp("2018-01-24 23:00"),
)

ZONES = [("核心区",), ("近港区",), ("外围区",)]
DIRECTIONS = [
    ("核心区", "近港区"), ("核心区", "外围区"),
    ("近港区", "核心区"), ("近港区", "外围区"),
    ("外围区", "核心区"), ("外围区", "近港区"),
]
VARIANTS = ("no_daily_count", "with_daily_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS, "all"],
        default="all",
        help="默认同时生成不含和包含每日拖轮总数的两组结果。",
    )
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args()


def load_processed(filename: str, group_cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DIR / filename,
        usecols=["time_window", *group_cols, "vessel_count"],
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )


def load_all_daily_counts() -> pd.DataFrame:
    training = pd.read_csv(
        PROCESSED_DIR / "daily_vessel_counts.csv",
        encoding="utf-8-sig",
        parse_dates=["date"],
    )
    validation = load_validation_count(
        RAW_DIR / "验证集_20180125-0131_每日拖轮数量.csv"
    )
    return (
        pd.concat([training, validation], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def mask_daily_counts(
    counts: pd.DataFrame,
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """让不可信标签日期的外生总数也不参与模型训练。"""
    result = counts.copy()
    dates = pd.to_datetime(result["date"]).dt.normalize()
    for start, end in ranges:
        invalid = dates.between(start.normalize(), end.normalize())
        result.loc[invalid, "vessel_count"] = pd.NA
    return result


def build_predictions(
    include_daily_count: bool,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_a = exclude_time_ranges(
        load_processed("task_a_train.csv", ["zone"]),
        [A_DATA_GAP],
    )
    task_b = exclude_time_ranges(
        load_processed("task_b_train.csv", ["source_zone", "target_zone"]),
        [B_DATA_GAP, B_TERMINAL_CENSORED],
    )
    daily_counts = load_all_daily_counts()
    forecast_times = pd.date_range(
        daily_counts["date"].max() - pd.Timedelta(days=6),
        periods=7 * 24,
        freq="h",
    )

    a_counts = mask_daily_counts(daily_counts, [A_DATA_GAP])
    b_counts = mask_daily_counts(daily_counts, [B_DATA_GAP])
    task_a_predictions = predict_pure_ml_daily(
        task_a,
        forecast_times,
        ["zone"],
        random_state=random_state,
        daily_vessel_counts=a_counts if include_daily_count else None,
    )
    task_b_predictions = predict_pure_ml_daily(
        task_b,
        forecast_times,
        ["source_zone", "target_zone"],
        random_state=random_state,
        daily_vessel_counts=b_counts if include_daily_count else None,
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
    args = parse_args()
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        include_daily_count = variant == "with_daily_count"
        print(f"Training LightGBM daily batch: {variant}...")
        task_a, task_b = build_predictions(include_daily_count, args.random_state)
        paths = generate_submissions(
            task_a,
            task_b,
            template_dir=RAW_DIR,
            output_dir=OUTPUT_ROOT / variant,
        )
        print(f"  Task A: {len(task_a)} rows, total={task_a['predicted'].sum()}")
        print(f"  Task B: {len(task_b)} rows, total={task_b['predicted'].sum()}")
        print(f"  Submission A: {paths[0]}")
        print(f"  Submission B: {paths[1]}")


if __name__ == "__main__":
    main()
