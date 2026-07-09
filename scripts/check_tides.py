"""自相关分析：观察潮汐周期对拖轮作业的影响。


结果很清楚，**潮汐漂移不明显**：

- **24h 峰值 0.42** — 日周期最强，没有往 25h 偏移
- **12h 峰值 0.32** — 半日周期也显著（早晚两个高峰，12.4h 被近似成了 12h）
- **48h 峰值 0.41** — 隔日相似度高
- **7d 相关为负** — 因为中间发生了天气事件，一周前和一周后完全不相似

**结论：** 50分钟的潮汐日漂移在这24天数据里被日周期覆盖了。
12h的半日潮周期才是潮汐的真实体现——一天两个作业高峰间隔约12小时，这个信号在0.32的相关系数里。
但预测时**用24h比用24h50min更稳**，因为50分钟漂移在24天尺度上还没拉开足够差距。

TODO: 其实0.42也不高，感觉小时的数不太考虑，以后可以分钟级试试。
"""
import pandas as pd
import numpy as np

# 加载 Task A 聚合数据，构建全局24天×24h的活跃拖轮总数序列
a = pd.read_csv(
    r"D:\Documents\PythonProject\CTS2026\data\processed\task_a_train.csv",
    parse_dates=["time_window"],
)

# 每小时总活跃拖轮数（三个圈层求和）
hourly_total = a.groupby("time_window")["vessel_count"].sum()
hourly_total = hourly_total.resample("h").sum().fillna(0)

values = hourly_total.values  # length = 576 (24 days × 24h)
n = len(values)

# 计算自相关（lag 1 到 200小时）
max_lag = 200
acf = []
for lag in range(1, max_lag + 1):
    corr = np.corrcoef(values[lag:], values[:-lag])[0, 1]
    acf.append(corr)

# 找出 lag 20~30 之间和 lag 140~170 之间的峰值
print("=" * 60)
print("Total active tug-hour autocorrelation (lag 1~200h)")
print("=" * 60)

# 详细看 lag 22~28 和 45~52 和 166~176
for name, lo, hi in [("~24h area", 21, 30), ("~48h area", 45, 52), ("~168h (7d) area", 140, 180)]:
    print(f"\n{name}:")
    for lag in range(lo, hi + 1):
        marker = " <--" if acf[lag - 1] == max(acf[lo - 1 : hi]) else ""
        print(f"  lag {lag:3d}h  ({lag/24:.2f}d): {acf[lag-1]:.4f}{marker}")

# 看 lag 24-26 之间有没有超过 24h 的偏移
print("\n" + "=" * 60)
print("Zoom: lag 22~27h")
print("=" * 60)
for lag in range(22, 28):
    print(f"  lag {lag}h ({lag/24:.2f}d): corr={acf[lag-1]:.4f}")

# 也检查半日潮对应的 ~12h 周期
print("\n" + "=" * 60)
print("Semi-diurnal check: lag 11~13h")
print("=" * 60)
for lag in range(11, 15):
    print(f"  lag {lag}h: corr={acf[lag-1]:.4f}")

# ==================== 分圈层看 ====================
for zone in a["zone"].unique():
    series = a[a["zone"] == zone].sort_values("time_window")["vessel_count"].values
    print(f"\n=== {zone} autocorrelation at key lags ===")
    for lag in [12, 24, 25, 48, 49]:
        c = np.corrcoef(series[lag:], series[:-lag])[0, 1]
        print(f"  lag {lag:3d}h: {c:.4f}")

print("\nDone.")
