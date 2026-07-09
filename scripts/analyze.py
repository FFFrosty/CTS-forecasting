"""数据分析：理解拖轮作业的周期性、稳定性和异常。"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 加载 ====================
a = pd.read_csv(
    r"D:\Documents\PythonProject\CTS2026\data\processed\task_a_train.csv",
    parse_dates=["time_window"],
)

ZONES = ["核心区", "近港区", "外围区"]

# 补齐完整的 index（含0值）
all_hours = pd.date_range("2018-01-01 00:00", "2018-01-24 23:00", freq="h")
idx = pd.MultiIndex.from_product([all_hours, ZONES], names=["time_window", "zone"])
a_full = a.set_index(["time_window", "zone"]).reindex(idx, fill_value=0).reset_index()

a_full["hour"] = a_full["time_window"].dt.hour
a_full["day_of_week"] = a_full["time_window"].dt.dayofweek  # 0=Mon, 6=Sun
a_full["date"] = a_full["time_window"].dt.date

# ==================== 图1：全时序 ====================
fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
for i, z in enumerate(ZONES):
    d = a_full[a_full["zone"] == z]
    axes[i].plot(d["time_window"], d["vessel_count"], linewidth=0.6, alpha=0.7)
    axes[i].set_ylabel(z)
    axes[i].grid(True, alpha=0.3)
axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig.suptitle("24天训练集——各圈层每小时活跃拖轮数", fontsize=13)
plt.tight_layout()
plt.savefig(r"D:\Documents\PythonProject\CTS2026\data\processed\fig_full_timeline.png", dpi=120)
plt.close()

# ==================== 图2：每周叠加 ====================
# 1月1日是周一，所以 week1=1-7, week2=8-14, week3=15-21, week4=22-24(部分)
a_full["week_label"] = "week" + ((pd.to_datetime(a_full["date"]) - pd.Timestamp("2018-01-01")).dt.days // 7 + 1).astype(str)
a_full["hour_of_week"] = a_full["day_of_week"] * 24 + a_full["hour"]

fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
for i, z in enumerate(ZONES):
    d = a_full[a_full["zone"] == z]
    for wl in d["week_label"].unique():
        w = d[d["week_label"] == wl]
        axes[i].plot(w["hour_of_week"], w["vessel_count"], linewidth=0.7, alpha=0.6, label=wl)
    axes[i].set_ylabel(z)
    axes[i].legend(fontsize=7, loc="upper right")
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel("hour of week (0=Mon 00:00)")
fig.suptitle("各周重叠对比——周模式稳定性", fontsize=13)
plt.tight_layout()
plt.savefig(r"D:\Documents\PythonProject\CTS2026\data\processed\fig_weekly_overlay.png", dpi=120)
plt.close()

# ==================== 图3：24小时×7天热力图 ====================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, z in enumerate(ZONES):
    piv = a_full[a_full["zone"] == z].pivot_table(
        index="hour", columns="day_of_week", values="vessel_count", aggfunc="mean"
    )
    im = axes[i].imshow(piv.values, aspect="auto", cmap="YlOrRd", origin="lower",
                        extent=[-0.5, 6.5, -0.5, 23.5])
    axes[i].set_title(z, fontsize=11)
    axes[i].set_xlabel("day of week")
    axes[i].set_xticks(range(7))
    axes[i].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[i].set_ylabel("hour")
    plt.colorbar(im, ax=axes[i], shrink=0.8)
fig.suptitle("小时×星期均值热力图 —— 什么时段最忙？", fontsize=13)
plt.tight_layout()
plt.savefig(r"D:\Documents\PythonProject\CTS2026\data\processed\fig_heatmap.png", dpi=120)
plt.close()

# ==================== 图4：每小时的波动范围 ====================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, z in enumerate(ZONES):
    d = a_full[a_full["zone"] == z]
    stats = d.groupby("hour")["vessel_count"].agg(["mean", "std", "min", "max"])
    axes[i].fill_between(stats.index, stats["min"], stats["max"], alpha=0.15, color="steelblue")
    axes[i].fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                         alpha=0.3, color="steelblue")
    axes[i].plot(stats.index, stats["mean"], color="steelblue", linewidth=2, label="mean")
    axes[i].set_title(z)
    axes[i].set_xlabel("hour")
    axes[i].set_ylabel("vessel_count")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()
fig.suptitle("各圈层24小时模式——均值±1σ+极值范围", fontsize=13)
plt.tight_layout()
plt.savefig(r"D:\Documents\PythonProject\CTS2026\data\processed\fig_hourly_pattern.png", dpi=120)
plt.close()

# ==================== 每日总数趋势 ====================
daily = a_full.groupby("date")["vessel_count"].sum().reset_index()
daily["dow"] = pd.to_datetime(daily["date"]).dt.dayofweek

print("=" * 50)
print("每日总活跃拖轮·小时数")
print("=" * 50)
for _, r in daily.iterrows():
    print(f"  {r['date']} (day {r['dow']}): {r['vessel_count']:.0f}")

print(f"\n总均值: {daily['vessel_count'].mean():.0f}")
print(f"标准差: {daily['vessel_count'].std():.0f}")
print(f"CV: {daily['vessel_count'].std() / daily['vessel_count'].mean():.2%}")

# 按星期几分组
daily["day_label"] = daily["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
print("\n按星期几均值:")
print(daily.groupby("day_label")["vessel_count"].agg(["mean", "std", "count"]))
