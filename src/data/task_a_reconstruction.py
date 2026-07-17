"""A题低采样时段的辅助数据源校准与伪标签复原。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


A_LOW_COVERAGE_START = pd.Timestamp("2018-01-12 07:00")
A_LOW_COVERAGE_END = pd.Timestamp("2018-01-19 09:00")
SECONDARY_SOURCES = ("e_globe_daily", "f_globe_dynamic")
ZONE_BITS = {"核心区": 4, "近港区": 2, "外围区": 1}

RECONSTRUCTION_FEATURES = [
    "sparse_count",
    "sparse_total",
    "secondary_records",
    "secondary_vessels",
    "hourly_vessel_coverage",
    "daily_vessel_count",
    "active_vessels",
    "zone_daily_unique",
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
]


@dataclass
class FittedReconstructionModel:
    pipeline: Pipeline
    healthy_hourly_coverage: float


def is_low_coverage(time_window: pd.Series) -> pd.Series:
    """返回A题需要复原的精确小时掩码。"""
    return pd.to_datetime(time_window).between(
        A_LOW_COVERAGE_START,
        A_LOW_COVERAGE_END,
    )


def build_sparse_observations(
    secondary_rows: pd.DataFrame,
    min_active_records: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从辅助数据源构造低采样A标签及每小时覆盖特征。"""
    required = {"mmsi", "time", "sog", "zone"}
    missing = required.difference(secondary_rows.columns)
    if missing:
        raise ValueError(f"secondary_rows are missing columns: {sorted(missing)}")

    work = secondary_rows[["mmsi", "time", "sog", "zone"]].copy()
    work["time"] = pd.to_datetime(work["time"])
    work["time_window"] = work["time"].dt.floor("h")
    coverage = (
        work.groupby("time_window", as_index=False)
        .agg(
            secondary_records=("mmsi", "size"),
            secondary_vessels=("mmsi", "nunique"),
        )
    )

    active = work[
        work["sog"].between(2.0, 10.0)
        & work["zone"].isin(ZONE_BITS)
    ]
    qualified = (
        active.groupby(["mmsi", "time_window", "zone"])
        .size()
        .rename("active_rows")
        .reset_index()
    )
    qualified = qualified[qualified["active_rows"] >= min_active_records]
    sparse = (
        qualified.groupby(["time_window", "zone"], as_index=False)["mmsi"]
        .nunique()
        .rename(columns={"mmsi": "sparse_count"})
    )
    return sparse, coverage


def build_daily_activity_statistics(
    vessel_state: pd.DataFrame,
    daily_vessel_counts: pd.DataFrame,
) -> pd.DataFrame:
    """生成每日活跃唯一船数及各圈层活跃唯一船数。"""
    required = {"mmsi", "time_window", "zone_state", "is_active"}
    missing = required.difference(vessel_state.columns)
    if missing:
        raise ValueError(f"vessel_state is missing columns: {sorted(missing)}")

    state = vessel_state[list(required)].copy()
    state["time_window"] = pd.to_datetime(state["time_window"])
    state["date"] = state["time_window"].dt.normalize()
    active = (
        state[state["is_active"]]
        .groupby("date")["mmsi"]
        .nunique()
        .rename("active_vessels")
    )

    zone_rows = []
    for zone, bit in ZONE_BITS.items():
        counts = (
            state[(state["zone_state"].astype(int) & bit) > 0]
            .groupby("date")["mmsi"]
            .nunique()
            .rename("zone_daily_unique")
            .reset_index()
        )
        counts["zone"] = zone
        zone_rows.append(counts)
    zones = pd.concat(zone_rows, ignore_index=True)

    daily = daily_vessel_counts[["date", "vessel_count"]].copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.rename(columns={"vessel_count": "daily_vessel_count"})
    zone_grid = daily[["date"]].merge(
        pd.DataFrame({"zone": list(ZONE_BITS)}),
        how="cross",
    )
    result = zone_grid.merge(
        zones,
        on=["date", "zone"],
        how="left",
        validate="one_to_one",
    )
    result["zone_daily_unique"] = result["zone_daily_unique"].fillna(0)
    result = result.merge(daily, on="date", how="left", validate="many_to_one")
    result = result.merge(active, on="date", how="left", validate="many_to_one")
    if result[["daily_vessel_count", "active_vessels"]].isna().any().any():
        raise ValueError("daily activity statistics are incomplete")
    return result.sort_values(["date", "zone"]).reset_index(drop=True)


def build_reconstruction_features(
    task_a: pd.DataFrame,
    sparse_counts: pd.DataFrame,
    hourly_coverage: pd.DataFrame,
    daily_statistics: pd.DataFrame,
) -> pd.DataFrame:
    """将完整A标签、辅助数据源和日级统计对齐为复原训练表。"""
    required = {"time_window", "zone", "vessel_count"}
    missing = required.difference(task_a.columns)
    if missing:
        raise ValueError(f"task_a is missing columns: {sorted(missing)}")

    frame = task_a[["time_window", "zone", "vessel_count"]].copy()
    frame["time_window"] = pd.to_datetime(frame["time_window"])
    frame = frame.rename(columns={"vessel_count": "original_vessel_count"})
    frame = frame.merge(
        sparse_counts,
        on=["time_window", "zone"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        hourly_coverage,
        on="time_window",
        how="left",
        validate="many_to_one",
    )
    frame[["sparse_count", "secondary_records", "secondary_vessels"]] = (
        frame[["sparse_count", "secondary_records", "secondary_vessels"]]
        .fillna(0.0)
    )
    frame["date"] = frame["time_window"].dt.normalize()
    frame = frame.merge(
        daily_statistics,
        on=["date", "zone"],
        how="left",
        validate="many_to_one",
    )
    if frame[[
        "daily_vessel_count",
        "active_vessels",
        "zone_daily_unique",
    ]].isna().any().any():
        raise ValueError("daily statistics do not cover the complete task A grid")

    frame["sparse_total"] = frame.groupby("time_window")["sparse_count"].transform(
        "sum"
    )
    frame["hourly_vessel_coverage"] = np.where(
        frame["daily_vessel_count"] > 0,
        frame["secondary_vessels"] / frame["daily_vessel_count"],
        0.0,
    )
    frame["hour"] = frame["time_window"].dt.hour
    frame["day_of_week"] = frame["time_window"].dt.dayofweek
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    return frame.sort_values(["time_window", "zone"]).reset_index(drop=True)


def _build_pipeline(random_state: int) -> Pipeline:
    categorical = Pipeline([
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    transformer = ColumnTransformer(
        [
            ("zone", categorical, ["zone"]),
            ("numeric", numeric, RECONSTRUCTION_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    ).set_output(transform="pandas")
    regressor = LGBMRegressor(
        objective="regression_l2",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("features", transformer), ("regressor", regressor)])


def fit_reconstruction_model(
    features: pd.DataFrame,
    random_state: int = 2026,
) -> FittedReconstructionModel:
    """在完整覆盖小时拟合辅助数据源到官方A标签的校准模型。"""
    low_coverage = is_low_coverage(features["time_window"])
    healthy = features[~low_coverage].copy()
    healthy_dates = healthy[
        ~healthy["date"].isin([
            A_LOW_COVERAGE_START.normalize(),
            A_LOW_COVERAGE_END.normalize(),
        ])
    ]
    if len(healthy) < 100:
        raise ValueError("at least 100 healthy task A rows are required")

    pipeline = _build_pipeline(random_state)
    pipeline.fit(
        healthy[["zone", *RECONSTRUCTION_FEATURES]],
        healthy["original_vessel_count"].astype(float),
    )
    healthy_hourly_coverage = float(
        healthy_dates["hourly_vessel_coverage"].median()
    )
    return FittedReconstructionModel(
        pipeline=pipeline,
        healthy_hourly_coverage=healthy_hourly_coverage,
    )


def reconstruct_task_a(
    features: pd.DataFrame,
    random_state: int = 2026,
    pseudo_label_weight: float = 0.35,
) -> pd.DataFrame:
    """复原异常小时，并保留原始标签、置信度和训练权重。"""
    if not 0.0 <= pseudo_label_weight <= 1.0:
        raise ValueError("pseudo_label_weight must be between 0 and 1")
    fitted = fit_reconstruction_model(features, random_state=random_state)
    result = features[[
        "time_window",
        "zone",
        "date",
        "original_vessel_count",
        "hourly_vessel_coverage",
    ]].copy()
    result["is_imputed"] = is_low_coverage(result["time_window"])
    result["reconstructed_count"] = result["original_vessel_count"].astype(float)

    gap = features[result["is_imputed"]]
    raw_predictions = np.clip(
        fitted.pipeline.predict(gap[["zone", *RECONSTRUCTION_FEATURES]]),
        0.0,
        None,
    )
    result.loc[result["is_imputed"], "reconstructed_count"] = raw_predictions

    denominator = max(fitted.healthy_hourly_coverage, 1e-6)
    coverage_confidence = (
        result["hourly_vessel_coverage"] / denominator
    ).clip(lower=0.1, upper=1.0)
    result["reconstruction_confidence"] = np.where(
        result["is_imputed"],
        coverage_confidence,
        1.0,
    )
    result["sample_weight"] = np.where(
        result["is_imputed"],
        pseudo_label_weight * result["reconstruction_confidence"],
        1.0,
    )
    result["vessel_count"] = result["reconstructed_count"]
    return result[[
        "time_window",
        "zone",
        "vessel_count",
        "original_vessel_count",
        "is_imputed",
        "reconstruction_confidence",
        "sample_weight",
    ]].sort_values(["time_window", "zone"]).reset_index(drop=True)


def cross_validate_reconstruction(
    features: pd.DataFrame,
    random_state: int = 2026,
) -> pd.DataFrame:
    """在完整日期上逐日留一，评估辅助数据源校准和日总量约束。"""
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
        pipeline = _build_pipeline(random_state)
        pipeline.fit(
            train[["zone", *RECONSTRUCTION_FEATURES]],
            train["original_vessel_count"].astype(float),
        )
        test["prediction_raw"] = np.clip(
            pipeline.predict(test[["zone", *RECONSTRUCTION_FEATURES]]),
            0.0,
            None,
        )

        daily_train = (
            train.groupby("date", as_index=False)
            .agg(
                task_a_total=("original_vessel_count", "sum"),
                active_vessels=("active_vessels", "first"),
            )
        )
        ratio = (
            daily_train["task_a_total"] / daily_train["active_vessels"]
        ).median()
        expected_total = float(test["active_vessels"].iloc[0]) * ratio
        raw_total = test["prediction_raw"].sum()
        scale = expected_total / raw_total if raw_total > 0 else 1.0
        test["prediction_constrained"] = test["prediction_raw"] * scale
        predictions.append(test[[
            "time_window",
            "date",
            "zone",
            "original_vessel_count",
            "prediction_raw",
            "prediction_constrained",
        ]])
    return pd.concat(predictions, ignore_index=True)
