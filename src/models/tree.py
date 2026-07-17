"""LightGBM 与 Random Forest 的 PureML 直接计数回归。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.ml import (
    build_causal_features,
    build_daily_batch_features,
    build_point_features,
    complete_time_grid,
    daily_batch_numeric_feature_columns,
    numeric_feature_columns,
)


MODEL_NAMES = ("lightgbm", "random_forest")


@dataclass
class FittedTreeModel:
    """保存拟合管道及其稳定特征约定。"""

    pipeline: Pipeline
    group_cols: tuple[str, ...]
    numeric_features: tuple[str, ...]
    model_name: str
    history_col: str

    @property
    def input_columns(self) -> list[str]:
        return [*self.group_cols, *self.numeric_features]


def _build_regressor(model_name: str, random_state: int):
    if model_name == "lightgbm":
        return LGBMRegressor(
            objective="regression_l2",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=20,
            reg_lambda=1.0,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"model_name must be one of {MODEL_NAMES}")


def fit_tree_model(
    train_samples: pd.DataFrame,
    group_cols: list[str],
    model_name: str,
    random_state: int = 2026,
    target_col: str = "vessel_count",
) -> FittedTreeModel:
    """用有效标签直接拟合计数回归模型。"""
    feature_frame = build_causal_features(
        train_samples,
        group_cols,
        target_col=target_col,
    )
    numeric_features = numeric_feature_columns()
    return _fit_feature_frame(
        feature_frame,
        group_cols,
        numeric_features,
        model_name,
        random_state,
        target_col,
        history_col=target_col,
    )


def _fit_feature_frame(
    feature_frame: pd.DataFrame,
    group_cols: list[str],
    numeric_features: list[str],
    model_name: str,
    random_state: int,
    target_col: str,
    history_col: str,
    sample_weights: pd.Series | None = None,
) -> FittedTreeModel:
    valid = feature_frame[target_col].notna()
    if valid.sum() < 10:
        raise ValueError("at least 10 labeled samples are required")

    categorical = Pipeline([
        (
            "one_hot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        )
    ])
    numeric = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        )
    ])
    transformer = ColumnTransformer(
        [
            ("categorical", categorical, group_cols),
            ("numeric", numeric, numeric_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    ).set_output(transform="pandas")
    pipeline = Pipeline([
        ("features", transformer),
        ("regressor", _build_regressor(model_name, random_state)),
    ])
    input_columns = [*group_cols, *numeric_features]
    fit_kwargs = {}
    if sample_weights is not None:
        weights = sample_weights.loc[valid].astype(float)
        if (
            not np.isfinite(weights).all()
            or weights.lt(0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "sample weights must be finite, non-negative and non-zero"
            )
        fit_kwargs["regressor__sample_weight"] = weights
    pipeline.fit(
        feature_frame.loc[valid, input_columns],
        feature_frame.loc[valid, target_col].astype(float),
        **fit_kwargs,
    )
    return FittedTreeModel(
        pipeline=pipeline,
        group_cols=tuple(group_cols),
        numeric_features=tuple(numeric_features),
        model_name=model_name,
        history_col=history_col,
    )


def fit_daily_batch_tree_model(
    train_samples: pd.DataFrame,
    group_cols: list[str],
    model_name: str = "lightgbm",
    random_state: int = 2026,
    daily_vessel_counts: pd.DataFrame | None = None,
    target_col: str = "vessel_count",
    history_col: str | None = None,
    sample_weight_col: str | None = None,
) -> FittedTreeModel:
    """拟合仅使用目标日开始前历史的每日批量树模型。"""
    history_col = history_col or target_col
    feature_frame = build_daily_batch_features(
        train_samples,
        group_cols,
        target_col=target_col,
        history_col=history_col,
        daily_vessel_counts=daily_vessel_counts,
    )
    sample_weights = None
    if sample_weight_col is not None:
        keys = ["time_window", *group_cols]
        required = {*keys, sample_weight_col}
        missing = required.difference(train_samples.columns)
        if missing:
            raise ValueError(
                f"train_samples are missing weight columns: {sorted(missing)}"
            )
        weights = train_samples[[*keys, sample_weight_col]].copy()
        if weights.duplicated(keys).any():
            raise ValueError("sample weights must contain unique time/group keys")
        feature_frame = feature_frame.merge(
            weights,
            on=keys,
            how="left",
            validate="one_to_one",
        )
        sample_weights = feature_frame[sample_weight_col]
    numeric_features = daily_batch_numeric_feature_columns(
        include_daily_count=daily_vessel_counts is not None,
    )
    return _fit_feature_frame(
        feature_frame,
        group_cols,
        numeric_features,
        model_name,
        random_state,
        target_col,
        history_col=history_col,
        sample_weights=sample_weights,
    )


def recursive_tree_forecast(
    fitted: FittedTreeModel,
    train_samples: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    target_col: str = "vessel_count",
) -> pd.DataFrame:
    """逐小时回填浮点预测，并在完整预测结束后转换为非负整数。"""
    times = pd.DatetimeIndex(pd.to_datetime(forecast_times))
    if times.empty or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("forecast_times must be non-empty, unique and sorted")
    if len(times) > 1 and not (times[1:] - times[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("forecast_times must be contiguous hourly timestamps")

    group_cols = list(fitted.group_cols)
    history_col = fitted.history_col
    history = complete_time_grid(
        train_samples,
        group_cols,
        target_col,
        extra_cols=[history_col] if history_col != target_col else [],
    )
    if times[0] <= history["time_window"].max():
        raise ValueError("forecast_times must start after the training history")
    origin = history["time_window"].min()
    groups = history[group_cols].drop_duplicates().sort_values(group_cols)

    full_index = pd.date_range(origin, times[-1], freq="h")
    histories: dict[tuple, pd.Series] = {}
    for group_values in groups.itertuples(index=False, name=None):
        mask = pd.Series(True, index=history.index)
        for column, value in zip(group_cols, group_values):
            mask &= history[column].eq(value)
        series = history.loc[mask].set_index("time_window")[history_col]
        histories[group_values] = series.reindex(full_index).astype(float)

    prediction_rows = []
    for time_window in times:
        feature_rows = []
        ordered_groups = []
        for group_values in groups.itertuples(index=False, name=None):
            row = dict(zip(group_cols, group_values))
            row.update(
                build_point_features(
                    histories[group_values],
                    time_window,
                    origin,
                )
            )
            feature_rows.append(row)
            ordered_groups.append(group_values)

        features = pd.DataFrame(feature_rows)[fitted.input_columns]
        raw_predictions = np.clip(fitted.pipeline.predict(features), 0.0, None)
        for group_values, prediction in zip(ordered_groups, raw_predictions):
            histories[group_values].loc[time_window] = float(prediction)
            prediction_rows.append({
                "time_window": time_window,
                **dict(zip(group_cols, group_values)),
                "predicted": float(prediction),
            })

    result = pd.DataFrame(prediction_rows)
    result["predicted"] = (
        np.rint(result["predicted"]).clip(lower=0).astype(int)
    )
    return result[["time_window", *group_cols, "predicted"]]


def daily_batch_tree_forecast(
    fitted: FittedTreeModel,
    train_samples: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    daily_vessel_counts: pd.DataFrame | None = None,
    target_col: str = "vessel_count",
) -> pd.DataFrame:
    """一次预测一天的全部小时，再用该日浮点预测递归到下一天。"""
    times = pd.DatetimeIndex(pd.to_datetime(forecast_times))
    if times.empty or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("forecast_times must be non-empty, unique and sorted")
    expected_times = pd.date_range(
        times[0].normalize(),
        times[-1].normalize() + pd.Timedelta(hours=23),
        freq="h",
    )
    if not times.equals(expected_times):
        raise ValueError("forecast_times must contain complete contiguous days")

    uses_daily_count = "daily_vessel_count" in fitted.numeric_features
    if uses_daily_count and daily_vessel_counts is None:
        raise ValueError("daily_vessel_counts are required by the fitted model")
    if uses_daily_count:
        required_dates = set(times.normalize())
        available = daily_vessel_counts.copy()
        available["date"] = pd.to_datetime(available["date"]).dt.normalize()
        available_dates = set(
            available.loc[available["vessel_count"].notna(), "date"]
        )
        if not required_dates.issubset(available_dates):
            raise ValueError("daily_vessel_counts are missing forecast dates")

    group_cols = list(fitted.group_cols)
    history_col = fitted.history_col
    history = complete_time_grid(
        train_samples,
        group_cols,
        target_col,
        extra_cols=[history_col] if history_col != target_col else [],
    )
    if times[0] <= history["time_window"].max():
        raise ValueError("forecast_times must start after the training history")
    groups = history[group_cols].drop_duplicates().sort_values(group_cols)
    prediction_frames = []

    for date in times.normalize().unique():
        day_times = times[times.normalize() == date]
        future = pd.DataFrame({"time_window": day_times}).merge(
            groups,
            how="cross",
        )
        future[target_col] = np.nan
        if history_col != target_col:
            future[history_col] = np.nan
        extended = pd.concat([history, future], ignore_index=True)
        feature_frame = build_daily_batch_features(
            extended,
            group_cols,
            target_col=target_col,
            history_col=history_col,
            daily_vessel_counts=daily_vessel_counts if uses_daily_count else None,
        )
        current = (
            feature_frame[feature_frame["time_window"].isin(day_times)]
            .sort_values(["time_window", *group_cols])
            .copy()
        )
        expected_rows = len(day_times) * len(groups)
        if len(current) != expected_rows:
            raise ValueError("daily prediction grid is incomplete")

        raw_predictions = np.clip(
            fitted.pipeline.predict(current[fitted.input_columns]),
            0.0,
            None,
        )
        predictions = current[["time_window", *group_cols]].copy()
        predictions["predicted"] = raw_predictions.astype(float)
        prediction_frames.append(predictions)

        updates = predictions.rename(columns={"predicted": history_col})
        if history_col != target_col:
            updates[target_col] = np.nan
        history = pd.concat([history, updates], ignore_index=True)

    result = pd.concat(prediction_frames, ignore_index=True)
    result["predicted"] = np.rint(result["predicted"]).clip(lower=0).astype(int)
    return result[["time_window", *group_cols, "predicted"]]


def predict_pure_ml(
    train_samples: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    model_name: str,
    random_state: int = 2026,
) -> pd.DataFrame:
    """适配现有 Predictor 接口的 PureML 训练与递归预测入口。"""
    fitted = fit_tree_model(
        train_samples,
        group_cols,
        model_name=model_name,
        random_state=random_state,
    )
    return recursive_tree_forecast(fitted, train_samples, forecast_times)


def predict_pure_ml_daily(
    train_samples: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    model_name: str = "lightgbm",
    random_state: int = 2026,
    daily_vessel_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """训练每日批量模型并按天递归生成完整预测。"""
    fitted = fit_daily_batch_tree_model(
        train_samples,
        group_cols,
        model_name=model_name,
        random_state=random_state,
        daily_vessel_counts=daily_vessel_counts,
    )
    return daily_batch_tree_forecast(
        fitted,
        train_samples,
        forecast_times,
        daily_vessel_counts=daily_vessel_counts,
    )
