"""每条船的时序规律可视化。"""
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = r"D:\Documents\PythonProject\CTS2026\data\processed"

state = pd.read_csv(f"{OUT_DIR}/vessel_state.csv", parse_dates=["time_window"])
state["hour"] = state["time_window"].dt.hour
state["day_of_week"] = state["time_window"].dt.dayofweek
state["date"] = state["time_window"].dt.date

# ==================== 按活跃小时排序 ====================
active_rank = state.groupby("mmsi")["is_active"].sum().sort_values(ascending=False)

# ==================== 图1: 每艘船一周模式热力图 (24h × 7d) ====================
# 取活跃最多的一批船
top_n = min(40, len(active_rank))
top_mmsis = active_rank.head(top_n).index

# 每艘船在每个 (hour, dow) 的平均 zone_state
heat_data = []
for mmsi in top_mmsis:
    vs = state[state["mmsi"] == mmsi]
    for h in range(24):
        for d in range(7):
            subset = vs[(vs["hour"] == h) & (vs["day_of_week"] == d)]
            if len(subset) > 0:
                mean_state = subset["zone_state"].mean()
            else:
                mean_state = 0
            heat_data.append({"mmsi": mmsi, "hour": h, "dow": d, "mean_state": mean_state})

heat_df = pd.DataFrame(heat_data)
heat_piv = heat_df.pivot_table(index="mmsi", columns=["dow", "hour"], values="mean_state")

# 确保每艘船有24*7列
all_cols = pd.MultiIndex.from_product([range(7), range(24)], names=["dow", "hour"])
heat_piv = heat_piv.reindex(columns=all_cols, fill_value=0)
heat_piv = heat_piv.loc[active_rank.head(top_n).sort_values().index]  # 活跃度从低到高

fig, ax = plt.subplots(figsize=(24, max(8, top_n * 0.25)))
cmap = plt.get_cmap("RdYlGn", 8)
im = ax.imshow(heat_piv.values, aspect="auto", cmap=cmap, vmin=-0.5, vmax=7.5)

# 标注星期几
for d in range(7):
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.axvline(x=d * 24 - 0.5, color="white", linewidth=1.5, alpha=0.6)
    ax.text(d * 24 + 11.5, -1.5, dow_names[d], ha="center", fontsize=8, fontweight="bold")

ax.set_yticks(range(len(heat_piv)))
ax.set_yticklabels([str(m) for m in heat_piv.index], fontsize=7)
ax.set_xticks(range(0, 24 * 7, 6))
ax.set_xticklabels([f"{h % 24:02d}" for h in range(0, 24 * 7, 6)], fontsize=7)
ax.set_xlabel("hour of week (Mon 00 ~ Sun 23)")

# 图例
legend_patches = [
    mpatches.Patch(color=cmap(i / 7), label=f"{i:03b}" if i > 0 else "idle")
    for i in range(8)
]
ax.legend(handles=legend_patches, title="zone_state", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=8)
ax.set_title(f"Each vessel's average zone_state by hour × day-of-week (top {top_n} vessels)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_vessel_pattern_grid.png", dpi=120, bbox_inches="tight")
plt.close()

# ==================== 图2: 前6艘船的完整24天时间线 ====================
show_n = 6
fig, axes = plt.subplots(show_n, 1, figsize=(22, show_n * 1.5), sharex=True)

for i, mmsi in enumerate(active_rank.head(show_n).index):
    ax = axes[i]
    vs = state[state["mmsi"] == mmsi].sort_values("time_window")

    # zone_state 颜色：0=白, 1=蓝(外), 2=橙(近), 3=绿, 4=红(核), 5=紫, 6=棕, 7=黑
    colors_list = ["#f0f0f0", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#333333"]
    colors = [colors_list[int(s)] for s in vs["zone_state"].values]

    ax.bar(vs["time_window"], height=1, width=1/24, color=colors, edgecolor="none", linewidth=0)
    ax.set_ylabel(str(mmsi), rotation=0, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

axes[-1].set_xlabel("date")
fig.suptitle(f"Top {show_n} vessels — 24-day zone_state timeline", fontsize=11, y=0.99)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_vessel_timeline.png", dpi=120)
plt.close()

# ==================== 图3: 每艘船的自相关 (活跃状态) ====================
# 用前20艘活跃船，画自相关函数看周期性
show_acf = min(20, len(active_rank))
fig, axes = plt.subplots(show_acf // 2, 2, figsize=(14, show_acf * 0.8))
axes = axes.flatten()

for i, mmsi in enumerate(active_rank.head(show_acf).index):
    ax = axes[i]
    vs = state[state["mmsi"] == mmsi].sort_values("time_window")
    vs = vs.set_index("time_window")

    # 在完整24天时间轴上填0
    full_idx = pd.date_range("2018-01-01", "2018-01-24 23:00", freq="h")
    active_series = vs["is_active"].reindex(full_idx, fill_value=0).astype(int)

    # 自相关
    lags = min(170, len(active_series) - 1)
    acf = [1.0]
    for lag in range(1, lags):
        corr = active_series[lag:].corr(active_series[:-lag])
        acf.append(corr)

    ax.stem(range(lags), acf, linefmt="steelblue", markerfmt=" ", basefmt="gray")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_title(f"mmsi={mmsi}", fontsize=8)
    ax.set_ylim(-0.3, 1)
    ax.set_xlim(0, lags)
    # 标记 24h, 48h, 168h
    for marker_lag in [24, 48, 72, 96, 120, 144, 168]:
        if marker_lag < lags:
            ax.axvline(x=marker_lag, color="red", linestyle="--", linewidth=0.5, alpha=0.3)

fig.suptitle("Per-vessel autocorrelation of is_active (24h mark in red)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_vessel_acf.png", dpi=120)
plt.close()

# ==================== 图4: 各船每日活跃小时数折线图 ====================
top15 = active_rank.head(15).index
daily_active = state[state["mmsi"].isin(top15)].groupby(["mmsi", "date"])["is_active"].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
for mmsi in top15:
    vs = daily_active[daily_active["mmsi"] == mmsi]
    ax.plot(vs["date"], vs["is_active"], marker=".", linewidth=0.8, alpha=0.7, label=str(mmsi))

ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.set_xlabel("date")
ax.set_ylabel("active hours per day")
ax.set_title("Top 15 vessels — daily active hours")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_vessel_daily_active.png", dpi=120)
plt.close()

print("Done. Output images in data/processed/")
