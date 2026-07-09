"""自回归预测模型：ARIMA / SARIMA / VAR。"""
import pandas as pd
import numpy as np


def fit_arima_forecast(
    series: np.ndarray,
    forecast_horizon: int = 168,
    order: tuple[int, int, int] = (3, 1, 2),
) -> np.ndarray:
    """对单变量序列拟合 ARIMA 并预测。

    Parameters
    ----------
    series : np.ndarray
        训练序列（1D）。
    forecast_horizon : int
        预测步数。
    order : tuple
        (p, d, q)。

    Returns
    -------
    np.ndarray
        预测值，长度 forecast_horizon。
    """
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=forecast_horizon)
    return forecast


def fit_sarima_forecast(
    series: np.ndarray,
    forecast_horizon: int = 168,
    order: tuple = (2, 0, 1),
    seasonal_order: tuple = (1, 0, 1, 24),
) -> np.ndarray:
    """对单变量序列拟合 SARIMA 并预测。

    Parameters
    ----------
    seasonal_order : tuple
        (P, D, Q, s)，s=24 表示日周期。
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=forecast_horizon)
    return forecast


def forecast_all_zones_arima(
    train_samples: pd.DataFrame,
    forecast_horizon: int = 168,
    target_col: str = "vessel_count",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """对每个分组独立做 ARIMA 预测。

    Returns
    -------
    pd.DataFrame
        含 time_window, group_cols, predicted。
    """
    if group_cols is None:
        group_cols = ["zone"]

    val_start = pd.Timestamp("2018-01-25")
    time_windows = pd.date_range(val_start, periods=forecast_horizon, freq="1h")

    results = []
    for keys, group in train_samples.groupby(group_cols):
        series = group[target_col].values
        forecast = fit_arima_forecast(series, forecast_horizon)

        entry = pd.DataFrame({"time_window": time_windows, "predicted": forecast})
        if isinstance(keys, tuple):
            for i, col in enumerate(group_cols):
                entry[col] = keys[i]
        else:
            entry[group_cols[0]] = keys
        results.append(entry)

    return pd.concat(results, ignore_index=True)
