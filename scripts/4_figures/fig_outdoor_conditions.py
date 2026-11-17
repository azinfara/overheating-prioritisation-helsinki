"""
Paper figure: outdoor conditions during the monitoring period, summer 2021.
Reads final/outdoor_daily.csv; saves plots/fig_outdoor_conditions_2021.png.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.environ.get("OVERHEATING_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
OUT = os.path.join(ROOT, "plots", "fig_outdoor_conditions_2021.png")
INK, MUTED, GRID = "#1E2329", "#5B6570", "#E3E6EA"
MAX_C, MEAN_C, MIN_C, HW = "#7A1F1F", "#1F5F8B", "#8FB3CC", "#F4D9C6"

d = pd.read_csv(os.path.join(ROOT, "final", "outdoor_daily.csv"), parse_dates=["date"])
d = d[(d["date"] >= "2021-05-15") & (d["date"] <= "2021-08-31")]
HEATWAVES = [("2021-06-18", "2021-06-23", "Short heatwave\n18–23 June"),
             ("2021-07-03", "2021-07-18", "Long heatwave\n3–18 July")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(6.7, 3.0))
for a, b, label in HEATWAVES:
    a, b = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1)
    ax.axvspan(a, b, color=HW, lw=0, zorder=0)
    ax.text(a + (b - a) / 2, 39.6, label, ha="center", va="top", fontsize=8, color=INK)
for y, label in ((25, "Hot day, 25 °C"), (30, "30 °C")):
    ax.axhline(y, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(pd.Timestamp("2021-08-31"), y + 0.4, label, ha="right", va="bottom", fontsize=7.5, color=MUTED)
for col, c, name in (("out_max", MAX_C, "Daily maximum"), ("out_mean", MEAN_C, "Daily mean"), ("out_min", MIN_C, "Daily minimum")):
    ax.plot(d["date"], d[col], color=c, lw=1.4, label=name, zorder=3)

ax.set_ylabel("Outdoor air temperature (°C)", color=INK)
ax.set_ylim(0, 40)
ax.set_yticks(range(0, 36, 5))
ax.set_xlim(pd.Timestamp("2021-05-15"), pd.Timestamp("2021-08-31"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(mdates.DateFormatter("1 %b"))
ax.tick_params(colors=MUTED, labelcolor=INK, length=3)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=300, facecolor="white")
print(os.path.abspath(OUT))
