"""统计预测的整数加权融合。"""
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


Predictor = Callable[[pd.DataFrame, pd.DatetimeIndex, list[str]], pd.DataFrame]


def predict_integer_blend(
    train_samples: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    group_cols: list[str],
    predictors: Sequence[Predictor],
    weights: Sequence[float],
) -> pd.DataFrame:
    """融合多个预测器，并在最后一步转换为非负整数。"""
    if len(predictors) == 0 or len(predictors) != len(weights):
        raise ValueError("predictors and weights must have the same non-zero length")
    if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("weights must be non-negative and sum to 1")

    keys = ["time_window"] + group_cols
    blended = None
    for index, (predictor, weight) in enumerate(zip(predictors, weights)):
        prediction = predictor(train_samples, forecast_times, group_cols)
        if prediction.duplicated(keys).any():
            raise ValueError("base predictions must have unique keys")
        prediction = prediction[keys + ["predicted"]].rename(
            columns={"predicted": f"prediction_{index}"}
        )
        if blended is None:
            blended = prediction
        else:
            blended = blended.merge(
                prediction,
                on=keys,
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            if not (blended["_merge"] == "both").all():
                raise ValueError("base predictions do not contain the same keys")
            blended = blended.drop(columns="_merge")
        blended[f"weighted_{index}"] = blended[f"prediction_{index}"] * weight

    weighted_cols = [f"weighted_{index}" for index in range(len(predictors))]
    if blended[weighted_cols].isna().any().any():
        raise ValueError("base predictions must not contain missing values")
    blended["predicted"] = (
        np.rint(blended[weighted_cols].sum(axis=1)).clip(lower=0).astype(int)
    )
    return blended[keys + ["predicted"]]
