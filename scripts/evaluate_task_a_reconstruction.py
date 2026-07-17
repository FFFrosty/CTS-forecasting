"""在正常日期模拟断档，比较A题伪标签复原方法。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.task_a_reconstruction import (
    A_LOW_COVERAGE_END,
    A_LOW_COVERAGE_START,
    cross_validate_reconstruction,
    is_low_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_profile_predictions(features: pd.DataFrame) -> pd.DataFrame:
    transition_dates = {
        A_LOW_COVERAGE_START.normalize(),
        A_LOW_COVERAGE_END.normalize(),
    }
    healthy = features[
        ~is_low_coverage(features["time_window"])
        & ~features["date"].isin(transition_dates)
    ].copy()
    predictions = []
    for date in sorted(healthy["date"].unique()):
        train = healthy[healthy["date"] != date]
        test = healthy[healthy["date"] == date].copy()
        exact = (
            train.groupby(["zone", "day_of_week", "hour"])[
                "original_vessel_count"
            ]
            .mean()
            .rename("prediction_profile")
        )
        fallback = (
            train.groupby(["zone", "hour"])["original_vessel_count"]
            .mean()
            .rename("fallback")
        )
        test = test.join(exact, on=["zone", "day_of_week", "hour"])
        test = test.join(fallback, on=["zone", "hour"])
        test["prediction_profile"] = test["prediction_profile"].fillna(
            test["fallback"]
        )
        predictions.append(test[[
            "time_window",
            "zone",
            "prediction_profile",
        ]])
    return pd.concat(predictions, ignore_index=True)


def summarize(predictions: pd.DataFrame, prediction_col: str) -> dict[str, float]:
    error = predictions[prediction_col] - predictions["original_vessel_count"]
    daily = predictions.groupby("date").agg(
        actual=("original_vessel_count", "sum"),
        predicted=(prediction_col, "sum"),
    )
    daily_error = daily["predicted"] - daily["actual"]
    return {
        "hourly_mae": float(error.abs().mean()),
        "hourly_rmse": float(np.sqrt(np.square(error).mean())),
        "daily_mae": float(daily_error.abs().mean()),
        "daily_rmse": float(np.sqrt(np.square(daily_error).mean())),
        "daily_max_abs": float(daily_error.abs().max()),
    }


def main() -> None:
    feature_path = PROCESSED_DIR / "task_a_reconstruction_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(
            "请先运行 scripts/reconstruct_task_a_gap.py 生成复原特征"
        )
    features = pd.read_csv(
        feature_path,
        encoding="utf-8-sig",
        parse_dates=["time_window", "date"],
    )
    calibrated = cross_validate_reconstruction(features)
    profile = build_profile_predictions(features)
    predictions = calibrated.merge(
        profile,
        on=["time_window", "zone"],
        how="left",
        validate="one_to_one",
    )

    summaries = []
    for name, column in [
        ("weekday_profile", "prediction_profile"),
        ("source_calibration", "prediction_raw"),
        ("source_plus_daily_constraint", "prediction_constrained"),
    ]:
        summaries.append({"method": name, **summarize(predictions, column)})
    summary = pd.DataFrame(summaries).sort_values("hourly_rmse")
    print(summary.to_string(
        index=False,
        formatters={
            column: "{:.3f}".format
            for column in summary.columns
            if column != "method"
        },
    ))

    output_path = PROCESSED_DIR / "task_a_reconstruction_cv.csv"
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Cross-validation rows: {output_path}")


if __name__ == "__main__":
    main()
