"""可视化：A 题训练数据的主要规律。

图1: (dow × hour) 活跃率热力图 —— 各时段有活跃记录的船数占比  
图2: 每船活跃模式聚类热力图 —— 选出勤天数≥14的船，看个体差异  
图3: 每日全局活跃总量 + 星期标注，一眼看趋势和异常期  
图4: Top 8 船每船一条 daily 柱状图，纵向排列，各自看个体规律  
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = PROCESSED_DIR / "viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

state = pd.read_csv(PROCESSED_DIR / "vessel_state.csv", parse_dates=["time_window"])
state["hour"] = state["time_window"].dt.hour
state["day_of_week"] = state["time_window"].dt.dayofweek  # 0=Mon
state["date"] = state["time_window"].dt.date

# 数据源断层期（1/13-1/18 主源 china_coastal 下线）
GAP_START = pd.Timestamp("2018-01-13")
GAP_END = pd.Timestamp("2018-01-18 23:59")
GAP_DATE_STRS = {str(d.date()) for d in pd.date_range("2018-01-13", "2018-01-18")}

# 排除断层期后的有效数据（用于图1/图2 的分母计算）
valid_mask = (state["time_window"] < GAP_START) | (state["time_window"] > GAP_END)
state_valid = state[valid_mask].copy()

# 动态计算有效期内每星期的天数（替代硬编码）
valid_dates = pd.Series(state_valid["date"].unique())
valid_dow_counts = valid_dates.apply(lambda d: d.weekday()).value_counts().sort_index()
total_days_per_dow = valid_dow_counts.rename("n_days")

# ==================== 通用：每船每小时是否“在港内活跃” ====================
# is_active 已经是窗口级标签，直接使用
active_rank = state.groupby("mmsi")["is_active"].sum().sort_values(ascending=False)

DOW_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# ==================== 图1: (dow × hour) 活跃率热力图 ====================
# 每个 (dow, hour) 有多少比例的天数有活跃记录（已排除断层期）
dow_hour_active = (
    state_valid.groupby(["day_of_week", "hour", "date"])["is_active"]
    .any().groupby(["day_of_week", "hour"]).sum()
    .rename("active_days").reset_index()
)
dow_hour_active = dow_hour_active.merge(
    total_days_per_dow, left_on="day_of_week", right_index=True
)
dow_hour_active["active_rate"] = dow_hour_active["active_days"] / dow_hour_active["n_days"]

heat = dow_hour_active.pivot(index="day_of_week", columns="hour", values="active_rate")
heat = heat.reindex(range(7))
heat = heat.reindex(columns=range(24))

fig, ax = plt.subplots(figsize=(16, 5))
im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

for d in range(7):
    for h in range(24):
        val = heat.iloc[d, h]
        ax.text(h, d, f"{val:.0%}" if val > 0 else "", ha="center", va="center", fontsize=7)

ax.set_yticks(range(7))
ax.set_yticklabels(DOW_NAMES)
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
ax.set_xlabel("小时")
ax.set_title("图1: 各时段有活跃记录的船数占比（已排除断层期1/13-1/18）")
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("活跃天数占比")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_dow_hour_active_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("图1 完成")


# ==================== 图2: 出勤天数≥14的船的 (dow × hour) 模式 ====================
attend_days = state.groupby("mmsi")["date"].nunique()
stable_mmsis = attend_days[attend_days >= 14].index
print(f"  出勤≥14天的船: {len(stable_mmsis)} 艘")

# 每船每 (dow, hour) 的活跃率（分母=该船该星期实际出勤天数，已排除断层期）
vessel_dow_days = (
    state_valid.groupby(["mmsi", "day_of_week"])["date"]
    .nunique().reset_index(name="n_days")
)

patterns = []
for mmsi in stable_mmsis:
    vs = state_valid[state_valid["mmsi"] == mmsi]
    dow_days_map = vessel_dow_days[vessel_dow_days["mmsi"] == mmsi].set_index("day_of_week")["n_days"]
    for d in range(7):
        vendor_days = dow_days_map.get(d, 0)
        for h in range(24):
            sub = vs[(vs["day_of_week"] == d) & (vs["hour"] == h)]
            active_count = sub["is_active"].sum()
            patterns.append({
                "mmsi": mmsi, "dow": d, "hour": h,
                "rate": active_count / vendor_days if vendor_days else 0
            })

pattern_df = pd.DataFrame(patterns)
piv = pattern_df.pivot_table(index="mmsi", columns=["dow", "hour"], values="rate")
all_cols = pd.MultiIndex.from_product([range(7), range(24)], names=["dow", "hour"])
piv = piv.reindex(columns=all_cols, fill_value=0)

# 按活跃总量排序
mmsi_order = piv.sum(axis=1).sort_values(ascending=False).index
piv = piv.loc[mmsi_order]

n_vessels = len(piv)
fig, ax = plt.subplots(figsize=(16, max(5, n_vessels * 0.3)))
im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

for d in range(7):
    ax.axvline(x=d * 24 - 0.5, color="white", linewidth=2)
    ax.text(d * 24 + 12, -1, DOW_NAMES[d], ha="center", fontsize=8, fontweight="bold")

ax.set_yticks(range(n_vessels))
ax.set_yticklabels([str(m) for m in piv.index], fontsize=8)
ax.set_xticks(range(0, 168, 6))
ax.set_xticklabels([f"{h % 24:02d}" for h in range(0, 168, 6)], fontsize=7)
ax.set_xlabel("小时（周一00:00 ~ 周日23:00）")
ax.set_title(f"图2: 出勤≥14天的 {n_vessels} 艘船 — 各时段活跃率（分母=每船实际出勤天数, 已排除断层期）")
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("活跃率")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_per_vessel_pattern.png", dpi=150, bbox_inches="tight")
plt.close()
print("图2 完成")


# ==================== 图3: 每日全局活跃总量 ====================
task_a = pd.read_csv(PROCESSED_DIR / "task_a_train.csv", parse_dates=["time_window"])
task_a["date"] = task_a["time_window"].dt.date
daily_total = task_a.groupby("date")["vessel_count"].sum().reset_index()
daily_total["dow_name"] = daily_total["date"].apply(
    lambda d: DOW_NAMES[d.weekday()]
)

fig, ax = plt.subplots(figsize=(14, 5))
dates = daily_total["date"].values
values = daily_total["vessel_count"].values
colors = ["#d62728" if str(d) in GAP_DATE_STRS
          else "#1f77b4" for d in daily_total["date"].astype(str)]
ax.bar(range(len(dates)), values, color=colors)

for i, (d, v) in enumerate(zip(dates, values)):
    day_name = daily_total["dow_name"].iloc[i]
    ax.text(i, v + 2, f"{v}", ha="center", fontsize=8)
    # 每周一天标记
    if day_name == "周一":
        ax.axvline(x=i - 0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

ax.set_xticks(range(len(dates)))
ax.set_xticklabels([str(d)[5:] for d in dates], rotation=45, fontsize=8)
ax.set_ylabel("每日总活跃量（vessel_count × 24h 求和）")

# 异常期标注（1/13-1/18，索引 12~17）
ax.axvspan(11.5, 17.5, color="red", alpha=0.08)
ax.text(14.5, ax.get_ylim()[1] * 0.95, "异常期:\n1/13-1/18", ha="center", fontsize=9, color="red")

ax.set_title("图3: 每日总活跃拖轮·小时数（红=异常期, 灰虚线=周一, 数字=当天总量）")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_daily_total.png", dpi=150, bbox_inches="tight")
plt.close()
print("图3 完成")


# ==================== 图4: Top 8 船每日活跃小时数 ====================
top8 = active_rank.head(8).index
daily_active = (
    state[state["mmsi"].isin(top8)]
    .groupby(["mmsi", "date"])["is_active"]
    .sum().reset_index()
    .rename(columns={"is_active": "active_hours"})
)
daily_active["day_of_week"] = daily_active["date"].apply(lambda d: d.weekday())

fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
axes = axes.flatten()

for i, mmsi in enumerate(top8):
    ax = axes[i]
    vs = daily_active[daily_active["mmsi"] == mmsi]
    dates = vs["date"].values
    hours = vs["active_hours"].values
    colors = ["#d62728" if str(d) in GAP_DATE_STRS
              else "#1f77b4" for d in dates]
    ax.bar(range(len(dates)), hours, color=colors)
    ax.text(len(dates) - 1, ax.get_ylim()[1] * 0.9, str(mmsi),
            ha="right", fontsize=9, fontweight="bold", color="#555555")
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.grid(axis="y", alpha=0.3)

    # 标出周末
    for j, d in enumerate(vs["day_of_week"].values):
        if d in [5, 6]:
            ax.axvspan(j - 0.5, j + 0.5, color="orange", alpha=0.08)

axes[-1].set_xticks(range(len(daily_active["date"].unique())))
axes[-1].set_xticklabels(
    [str(d)[5:10] for d in sorted(daily_active["date"].unique())],
    rotation=45, fontsize=8
)
fig.suptitle("图4: Top 8 活跃船 — 每日活跃小时数（蓝=正常, 红=异常期, 橙底色=周末）", y=0.99)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_top8_daily.png", dpi=150, bbox_inches="tight")
plt.close()
print("图4 完成")


# ==================== 图5: 每艘高频船的 (日期×小时) 区域+活跃叠加热力图 ====================
print("\n绘制图5: 每船日×时状态图...")
vessel_daily_dir = OUT_DIR / "vessel_daily"
vessel_daily_dir.mkdir(parents=True, exist_ok=True)

# 区域颜色映射
ZONE_CODE = {"无数据": 0, "港外": 1, "外围区": 2, "近港区": 3, "核心区": 4}
ZONE_COLORS = ["#ffffff", "#cccccc", "#fff176", "#ff9800", "#e53935"]
ZONE_LABELS = ["无数据", "港外", "外围区", "近港区", "核心区"]
cmap_zone = ListedColormap(ZONE_COLORS)
norm_zone = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)

# 完整日期×小时网格（含断层期，画全部数据）
all_dates = sorted(state["date"].unique())
all_hours = list(range(24))

# 断层期日期集合（用于背景标注）
gap_date_set = set(pd.date_range("2018-01-13", "2018-01-18").date)

for mmsi in stable_mmsis:
    vs = state[state["mmsi"] == mmsi].copy()
    # 构造 日期×小时 矩阵
    pivot_zone = vs.pivot_table(
        index="date", columns="hour", values="primary_zone",
        aggfunc="first"
    ).reindex(index=all_dates, columns=all_hours)
    pivot_active = vs.pivot_table(
        index="date", columns="hour", values="is_active",
        aggfunc="first"
    ).reindex(index=all_dates, columns=all_hours).eq(True)

    # zone 编码为整数
    zone_matrix = pivot_zone.map(lambda z: ZONE_CODE.get(z, 0) if pd.notna(z) else 0).values

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(zone_matrix, aspect="auto", cmap=cmap_zone, norm=norm_zone, interpolation="nearest")

    # 活跃格子加黑色边框
    for i in range(len(all_dates)):
        for j in range(24):
            if pivot_active.iloc[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="black", linewidth=1.5))

    # 周末 & 断层期背景标注
    for i, date in enumerate(all_dates):
        if date in gap_date_set:
            ax.axhspan(i - 0.5, i + 0.5, color="red", alpha=0.08, zorder=0)
        elif date.weekday() in [5, 6]:
            ax.axhspan(i - 0.5, i + 0.5, color="orange", alpha=0.05, zorder=0)

    # 轴标签
    ax.set_yticks(range(len(all_dates)))
    ax.set_yticklabels([f"{d.strftime('%m-%d')} {DOW_NAMES[d.weekday()]}" for d in all_dates], fontsize=8)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_xlabel("小时")
    ax.set_title(f"船 {mmsi} — 每日每小时区域分布（黑框=活跃作业, 红底=断层期, 橙底=周末）")

    # 图例
    legend_elements = [Patch(facecolor=ZONE_COLORS[i], label=ZONE_LABELS[i]) for i in range(5)]
    legend_elements.append(Patch(facecolor="white", edgecolor="black", linewidth=1.5, label="活跃作业"))
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(vessel_daily_dir / f"{mmsi}.png", dpi=120, bbox_inches="tight")
    plt.close()

print(f"图5 完成: {len(stable_mmsis)} 艘船 → {vessel_daily_dir}")

print(f"\n全部完成。图片输出: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("fig*.png")):
    print(f"  {f.name}")
print(f"  vessel_daily/ ({len(stable_mmsis)} 张船图)")
