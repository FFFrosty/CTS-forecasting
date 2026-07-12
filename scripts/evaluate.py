"""回测评估：在训练集上做滚动验证，评估不同策略的预测能力。

排除数据源断层期（1/13-1/18）后再评估，避免主源缺失导致的标签虚低污染。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

ZONES = ["核心区", "近港区", "外围区"]
DIRECTIONS = [
    ("核心区", "近港区"), ("核心区", "外围区"),
    ("近港区", "核心区"), ("近港区", "外围区"),
    ("外围区", "核心区"), ("外围区", "近港区"),
]

# 数据源断层期，排除
ANOM_START = pd.Timestamp("2018-01-13")
ANOM_END = pd.Timestamp("2018-01-18 23:00")


def load_and_filter(name: str, index_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        PROCESSED_DIR / name,
        encoding="utf-8-sig",
        parse_dates=["time_window"],
    )
    # 排除断层期
    df = df[
        (df["time_window"] < ANOM_START) | (df["time_window"] > ANOM_END)
    ].copy()
    # 时间特征
    df["hour"] = df["time_window"].dt.hour
    df["day_of_week"] = df["time_window"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def walk_forward_eval(
    df, group_cols, train_days, test_days, pred_fn, target="vessel_count"
):
    """滑动窗口评估：用 train_days 天训练，预测接下来 test_days 天。"""
    all_hours = np.sort(df["time_window"].unique())
    total_hours = len(all_hours)

    errors = []
    for start_day in range(total_hours // 24 - train_days - test_days + 1):
        train_end_h = (start_day + train_days) * 24
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

        preds = pred_fn(train_df, len(test_times), group_cols)
        test_sorted = test_df.sort_values(["time_window"] + group_cols)
        actuals = test_sorted[target].values

        if len(preds) != len(actuals):
            test_pivoted = test_df.pivot_table(
                index="time_window", columns=group_cols, values=target, fill_value=0
            )
            actuals = test_pivoted.values.flatten()
            if len(preds) != len(actuals):
                min_len = min(len(preds), len(actuals))
                preds = preds[:min_len]
                actuals = actuals[:min_len]

        mae = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        errors.append({
            "train_days": train_days,
            "test_start": test_times[0],
            "mae": mae,
            "rmse": rmse,
        })

    return pd.DataFrame(errors)


# ==================== 策略函数 ====================

def pred_historical_mean(train_df, forecast_hours, group_cols):
    """策略1：同星期同时刻历史均值。"""
    mean_by_hour_dow = train_df.groupby(
        group_cols + ["hour", "day_of_week"]
    )["vessel_count"].mean().reset_index(name="pred")

    last_time = train_df["time_window"].max()
    test_times = pd.date_range(
        last_time + pd.Timedelta(hours=1), periods=forecast_hours, freq="h"
    )

    preds = []
    for tw in test_times:
        matched = mean_by_hour_dow[
            (mean_by_hour_dow["hour"] == tw.hour)
            & (mean_by_hour_dow["day_of_week"] == tw.dayofweek)
        ].sort_values(group_cols)
        preds.append(matched["pred"].values)

    return np.array(preds).flatten()


def pred_last_n_mean(train_df, forecast_hours, group_cols, n=24):
    """策略2：训练集最后 n 小时均值→常数外推。"""
    df_sorted = train_df.sort_values("time_window")
    last_n = df_sorted.groupby(group_cols).tail(n)
    means = last_n.groupby(group_cols)["vessel_count"].mean()
    ref_idx = ZONES if len(group_cols) == 1 else DIRECTIONS
    means = means.reindex(ref_idx, fill_value=0)
    return np.tile(means.values, forecast_hours)


def pred_global_mean(train_df, forecast_hours, group_cols):
    """策略3：全历史均值常数。"""
    means = train_df.groupby(group_cols)["vessel_count"].mean()
    ref_idx = ZONES if len(group_cols) == 1 else DIRECTIONS
    means = means.reindex(ref_idx, fill_value=0)
    return np.tile(means.values, forecast_hours)


# ==================== 评估 ====================

def main():
    a_full = load_and_filter("task_a_train.csv", ["zone"])
    b_full = load_and_filter("task_b_train.csv", ["source_zone", "target_zone"])

    for label, df, group_cols in [
        ("赛题A", a_full, ["zone"]),
        ("赛题B", b_full, ["source_zone", "target_zone"]),
    ]:
        print("=" * 65)
        print(f"{label}：不同策略滑动窗口回测（已排除数据源断层期）")
        print("=" * 65)
        for train_days in [7, 14, 17]:
            print(f"\n--- 前 {train_days} 天训练，预测后续 ---")
            for name, fn in [
                ("同星期同时刻均值", pred_historical_mean),
                ("最后24h均值外推", pred_last_n_mean),
                ("全历史均值常数", pred_global_mean),
            ]:
                res = walk_forward_eval(df, group_cols, train_days, 3, fn)
                if len(res) > 0:
                    print(f"  {name}: MAE={res['mae'].mean():.3f}, RMSE={res['rmse'].mean():.3f}")


if __name__ == "__main__":
    main()
