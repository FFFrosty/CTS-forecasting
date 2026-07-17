"""训练使用 A 题重建历史的 PureML v6，并生成两组对照提交。"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_validation_count
from src.evaluation import exclude_time_ranges
from src.models.tree import (
    daily_batch_tree_forecast,
    fit_daily_batch_tree_model,
    predict_pure_ml_daily,
)
from src.submission import generate_submissions, validate_prediction_grid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "submission" / "PureML_v6" / "reconstructed"

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
VARIANTS = ("feature_only", "weighted_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS, "all"],
        default="all",
    )
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args()


def load_reconstructed_task_a(variant: str) -> pd.DataFrame:
    path = PROCESSED_DIR / "task_a_train_reconstructed.csv"
    if not path.exists():
        raise FileNotFoundError(
            "请先运行 scripts/reconstruct_task_a_gap.py 生成 A 题重建历史"
        )
    task_a = pd.read_csv(
        path,
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )
    task_a["history_count"] = task_a["vessel_count"].astype(float)
    is_imputed = task_a["is_imputed"]
    if not pd.api.types.is_bool_dtype(is_imputed):
        is_imputed = is_imputed.astype(str).str.lower().eq("true")

    if variant == "feature_only":
        task_a.loc[is_imputed, "vessel_count"] = pd.NA
        return task_a[[
            "time_window",
            "zone",
            "vessel_count",
            "history_count",
        ]]
    return task_a[[
        "time_window",
        "zone",
        "vessel_count",
        "history_count",
        "sample_weight",
    ]]


def load_task_b() -> pd.DataFrame:
    task_b = pd.read_csv(
        PROCESSED_DIR / "task_b_train.csv",
        usecols=[
            "time_window",
            "source_zone",
            "target_zone",
            "vessel_count",
        ],
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )
    return exclude_time_ranges(task_b, [B_DATA_GAP, B_TERMINAL_CENSORED])


def validation_times() -> pd.DatetimeIndex:
    validation_path = next(
        path
        for path in RAW_DIR.glob("*.csv")
        if "0131" in path.name and "每日拖轮数量" in path.name
    )
    counts = load_validation_count(validation_path)
    return pd.date_range(
        counts["date"].min(),
        counts["date"].max() + pd.Timedelta(hours=23),
        freq="h",
    )


def build_predictions(
    variant: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_a = load_reconstructed_task_a(variant)
    task_b = load_task_b()
    forecast_times = validation_times()

    fitted_a = fit_daily_batch_tree_model(
        task_a,
        ["zone"],
        random_state=random_state,
        history_col="history_count",
        sample_weight_col=(
            "sample_weight" if variant == "weighted_labels" else None
        ),
    )
    task_a_predictions = daily_batch_tree_forecast(
        fitted_a,
        task_a,
        forecast_times,
    )
    task_b_predictions = predict_pure_ml_daily(
        task_b,
        forecast_times,
        ["source_zone", "target_zone"],
        random_state=random_state,
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
        print(f"Training PureML v6: {variant}...")
        task_a, task_b = build_predictions(variant, args.random_state)
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
