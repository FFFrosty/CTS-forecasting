"""回测评估：在训练集上做滚动验证，评估不同策略的预测能力。"""
import pandas as pd
import numpy as np

# ==================== 加载数据 ====================
a = pd.read_csv(
    r"D:\Documents\PythonProject\CTS2026\data\processed\task_a_train.csv",
    parse_dates=["time_window"],
)
b = pd.read_csv(
    r"D:\Documents\PythonProject\CTS2026\data\processed\task_b_train.csv",
    parse_dates=["time_window"],
)

ZONES = ["核心区", "近港区", "外围区"]
DIRECTIONS = [
    ("核心区", "近港区"), ("核心区", "外围区"),
    ("近港区", "核心区"), ("近港区", "外围区"),
    ("外围区", "核心区"), ("外围区", "近港区"),
]


def expand_full_index(df, index_cols):
    """补齐缺失的零值行，确保所有时间窗口×分组都有记录。"""
    all_hours = pd.date_range("2018-01-01", "2018-01-25", freq="h", inclusive="left")
    if index_cols == ["zone"]:
        idx = pd.MultiIndex.from_product([all_hours, ZONES], names=["time_window"] + index_cols)
    else:
        idx = pd.MultiIndex.from_product(
            [all_hours, [d[0] for d in DIRECTIONS], [d[1] for d in DIRECTIONS]],
            names=["time_window"] + index_cols,
        )
    df_full = df.set_index(["time_window"] + index_cols).reindex(idx, fill_value=0).reset_index()
    return df_full


def add_time_info(df):
    """补齐时间特征。"""
    df["hour"] = df["time_window"].dt.hour
    df["day_of_week"] = df["time_window"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


# 补齐零值
a_full = expand_full_index(a, ["zone"])
a_full = add_time_info(a_full)

b_full = expand_full_index(b, ["source_zone", "target_zone"])
b_full = add_time_info(b_full)


def walk_forward_eval(df, group_cols, train_days, test_days, pred_fn, target="vessel_count"):
    """滑动窗口评估：用 train_days 天的数据训练策略，预测接下来 test_days 天。

    Parameters
    ----------
    pred_fn : callable(train_df, forecast_hours, group_cols) -> np.ndarray
        预测函数，返回 shape (forecast_hours, n_groups) 的预测值。
    """
    all_hours = df["time_window"].unique()
    all_hours = np.sort(all_hours)
    total_hours = len(all_hours)
    step = 24  # 每天滑一步

    errors = []
    for start_day in range(0, train_days, 1):
        train_end_h = start_day * 24 + train_days * 24
        test_start_h = train_end_h
        test_end_h = test_start_h + test_days * 24
        if test_end_h > total_hours:
            break

        train_times = all_hours[:train_end_h]
        test_times = all_hours[test_start_h:test_end_h]

        train_df = df[df["time_window"].isin(train_times)]
        test_df = df[df["time_window"].isin(test_times)]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # 预测
        forecast_hours = len(test_times)
        preds = pred_fn(train_df, forecast_hours, group_cols)

        # 真实值
        test_sorted = test_df.sort_values(["time_window"] + group_cols)
        actuals = test_sorted[target].values

        # 对齐预测和真实值
        if len(preds) != len(actuals):
            # 对齐分组顺序
            test_pivoted = test_df.pivot_table(
                index="time_window", columns=group_cols, values=target, fill_value=0
            )
            actuals = test_pivoted.values.flatten()
            if len(preds) != len(actuals):
                min_len = min(len(preds), len(actuals))
                preds = preds[:min_len]
                actuals = actuals[:min_len]

        mae = np.mean(np.abs(preds - actuals))
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        sse = np.sum((preds - actuals) ** 2)
        errors.append({"train_days": train_days, "test_start": test_times[0], "mae": mae, "rmse": rmse, "sse": sse, "n_actual": len(actuals)})

    return pd.DataFrame(errors)


def pred_historical_mean(train_df, forecast_hours, group_cols):
    """策略1：同星期同时刻历史均值。"""
    mean_by_hour_dow = train_df.groupby(group_cols + ["hour", "day_of_week"])["vessel_count"].mean()
    mean_by_hour_dow = mean_by_hour_dow.reset_index(name="pred")

    # 生成预测序列：从 train 最后一小时之后开始
    last_time = train_df["time_window"].max()
    test_times = pd.date_range(last_time + pd.Timedelta(hours=1), periods=forecast_hours, freq="h")

    preds = []
    for tw in test_times:
        matched = mean_by_hour_dow[
            (mean_by_hour_dow["hour"] == tw.hour)
            & (mean_by_hour_dow["day_of_week"] == tw.dayofweek)
        ]
        matched = matched.sort_values(group_cols)
        preds.append(matched["pred"].values)

    return np.array(preds).flatten()


def pred_last_n_mean(train_df, forecast_hours, group_cols, n=24):
    """策略2：训练集最后 n 小时的各分组均值，作为常数外推。"""
    df_sorted = train_df.sort_values("time_window")
    last_n = df_sorted.groupby(group_cols).tail(n)
    means = last_n.groupby(group_cols)["vessel_count"].mean()
    # 排序确保顺序一致
    if len(group_cols) == 1:
        means = means.reindex(ZONES, fill_value=0)
    else:
        idx = pd.MultiIndex.from_tuples(DIRECTIONS)
        means = means.reindex(idx, fill_value=0)
    pred = means.values
    return np.tile(pred, forecast_hours)


def pred_global_mean(train_df, forecast_hours, group_cols):
    """策略3：所有历史数据的各分组均值，纯常数。"""
    means = train_df.groupby(group_cols)["vessel_count"].mean()
    if len(group_cols) == 1:
        means = means.reindex(ZONES, fill_value=0)
    else:
        idx = pd.MultiIndex.from_tuples(DIRECTIONS)
        means = means.reindex(idx, fill_value=0)
    pred = means.values
    return np.tile(pred, forecast_hours)


# ==================== 评估 ====================
print("=" * 60)
print("赛题A：不同策略滑动窗口回测")
print("=" * 60)

for train_days in [7, 14, 17]:
    print(f"\n--- 前 {train_days} 天训练，预测后续 ---")
    for name, fn in [
        ("历史同星期同时刻均值", pred_historical_mean),
        ("最后24h均值外推", pred_last_n_mean),
        ("全历史均值常数", pred_global_mean),
    ]:
        res = walk_forward_eval(a_full, ["zone"], train_days, 3, fn)
        if len(res) > 0:
            print(f"  {name}: MAE={res['mae'].mean():.3f}, RMSE={res['rmse'].mean():.3f}")

print()
print("=" * 60)
print("赛题B：不同策略滑动窗口回测")
print("=" * 60)

for train_days in [7, 14, 17]:
    print(f"\n--- 前 {train_days} 天训练，预测后续 ---")
    for name, fn in [
        ("历史同星期同时刻均值", pred_historical_mean),
        ("最后24h均值外推", pred_last_n_mean),
        ("全历史均值常数", pred_global_mean),
    ]:
        res = walk_forward_eval(b_full, ["source_zone", "target_zone"], train_days, 3, fn)
        if len(res) > 0:
            print(f"  {name}: MAE={res['mae'].mean():.3f}, RMSE={res['rmse'].mean():.3f}")
