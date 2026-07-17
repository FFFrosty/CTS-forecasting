"""训练 PureML v5，并生成 LightGBM/RF 的 A、B 提交文件。"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_validation_count
from src.evaluation import exclude_time_ranges
from src.models.tree import MODEL_NAMES, predict_pure_ml
from src.submission import generate_submissions, validate_prediction_grid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "submission" / "PureML_v5"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[*MODEL_NAMES, "all"],
        default="all",
        help="要训练的模型；默认同时生成 LightGBM 和 Random Forest。",
    )
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args()


def load_processed(filename: str, group_cols: list[str]) -> pd.DataFrame:
    columns = ["time_window", *group_cols, "vessel_count"]
    return pd.read_csv(
        PROCESSED_DIR / filename,
        usecols=columns,
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )


def build_predictions(
    model_name: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    task_a = exclude_time_ranges(
        load_processed("task_a_train.csv", ["zone"]),
        [A_DATA_GAP],
    )
    task_b = exclude_time_ranges(
        load_processed("task_b_train.csv", ["source_zone", "target_zone"]),
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

    task_a_predictions = predict_pure_ml(
        task_a,
        forecast_times,
        ["zone"],
        model_name=model_name,
        random_state=random_state,
    )
    task_b_predictions = predict_pure_ml(
        task_b,
        forecast_times,
        ["source_zone", "target_zone"],
        model_name=model_name,
        random_state=random_state,
    )
    validate_prediction_grid(task_a_predictions, forecast_times, ["zone"], ZONES)
    validate_prediction_grid(
        task_b_predictions,
        forecast_times,
        ["source_zone", "target_zone"],
        DIRECTIONS,
    )
    return task_a_predictions, task_b_predictions, forecast_times


def main() -> None:
    args = parse_args()
    model_names = MODEL_NAMES if args.model == "all" else (args.model,)
    for model_name in model_names:
        print(f"Training {model_name}...")
        task_a, task_b, _ = build_predictions(model_name, args.random_state)
        paths = generate_submissions(
            task_a,
            task_b,
            template_dir=RAW_DIR,
            output_dir=OUTPUT_ROOT / model_name,
        )
        print(f"  Task A: {len(task_a)} rows, total={task_a['predicted'].sum()}")
        print(f"  Task B: {len(task_b)} rows, total={task_b['predicted'].sum()}")
        print(f"  Submission A: {paths[0]}")
        print(f"  Submission B: {paths[1]}")


if __name__ == "__main__":
    main()
