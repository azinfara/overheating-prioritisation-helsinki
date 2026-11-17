"""
Results figures and the risk-group table of the paper (added 13 November 2026).

Saves (figure numbers as in the manuscript)
  exports/table_risk_groups.csv               Table 4, Section 3.1
  plots/fig3_within_building_variation.png   Fig. 3, Section 3.2 (two panels stacked vertically)
  plots/fig4_variance_and_predictability.png Fig. 4, Section 3.3
  plots/fig5_prioritisation.png              Fig. 5, Section 3.6

Reads exports/step2_r2.csv (stage 6), exports/table4_r2.csv (stage 13) and
exports/oof_point_scores.csv (stage 14), so run those stages first.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from common import (DATA, PLOTS, MIN_APTS, DG_ORDER, load_buildings_metadata, load_apartment_metrics,
                    build_severity_cluster, build_composition_cluster)

EXPORTS = os.path.join(os.path.dirname(DATA), "exports")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"                      # categorical slots 1-2
RISK = {"Low": "#f6c7ae", "Medium": "#eb6834", "High": "#9e3a10"}   # one hue, light -> dark
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": "#b9b8b3",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def save(fig, name):
    p = os.path.join(PLOTS, name)
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)


meta = load_buildings_metadata()
apt = load_apartment_metrics(merge_area_type=False)
apt = apt.merge(meta[["building_id", "design_group"]], on="building_id", how="left")
n_apt = apt.groupby("building_id")["device"].count()
b3 = n_apt[n_apt >= MIN_APTS].index

# ---------------------------------------------------------------- targets
bc = (apt[apt["building_id"].isin(b3)].groupby("building_id")["dh_27"]
      .apply(lambda v: (v > 150).mean()).rename("frac_over").to_frame())
ta, _ = build_composition_cluster(bc, min_apts=MIN_APTS)
tb, _ = build_severity_cluster(meta, min_apts=MIN_APTS)
m = meta.set_index("building_id")
prof_cols = {"bld_hw_temp_mean": "Mean heatwave temperature (°C)", "bld_hw_temp_max": "Mean heatwave maximum (°C)",
             "bld_dh_27_mean": "Degree-hours above 27 °C (Kh)", "bld_dh_30_mean": "Degree-hours above 30 °C (Kh)"}
rows = []
allb = m.loc[ta.index, list(prof_cols)].join(bc)
rows.append({"Target": "All", "Group": "All buildings", "Buildings": len(allb),
             "Share of overheated apartments (%)": f"{allb['frac_over'].min()*100:.0f}–{allb['frac_over'].max()*100:.0f}",
             **{v: round(allb[k].mean(), 1 if "°C)" in v else 0) for k, v in prof_cols.items()}})
for name, t in (("A", ta), ("B", tb)):
    d = m.loc[t.index, list(prof_cols)].join(bc).assign(tier=t["cluster_tier"])
    for tier in ["Low", "Medium", "High"]:
        g = d[d["tier"] == tier]
        rows.append({"Target": name, "Group": tier, "Buildings": len(g),
                     "Share of overheated apartments (%)": f"{g['frac_over'].min()*100:.0f}–{g['frac_over'].max()*100:.0f}",
                     **{v: round(g[k].mean(), 1 if "°C)" in v else 0) for k, v in prof_cols.items()}})
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(EXPORTS, "table_risk_groups.csv"), index=False, encoding="utf-8-sig")
print(tab.to_string(index=False))

# ---------------------------------------------------------------- Fig 3: within-building variation (vertical)
rng = np.random.default_rng(42)
rws = {}
for b, g in apt[apt["building_id"].isin(b3)].dropna(subset=["temp_mean"]).groupby("building_id"):
    x = g["temp_mean"].to_numpy()
    rws[b] = {"n": len(x), "raw": np.ptp(x), "three": np.mean([np.ptp(rng.choice(x, 3, replace=False)) for _ in range(500)])}
s = pd.DataFrame.from_dict(rws, orient="index")
s["grp"] = pd.cut(s["n"], [2, 5, 9, 14, 999], labels=["3–5", "6–9", "10–14", "≥ 15"])
FS, FT = 10, 11          # label and title font sizes (figure is placed at about 13 cm width)
BOXKW = dict(showfliers=False, patch_artist=True, medianprops={"color": "white", "lw": 1.6})
def boxes(a, data, pos, c, w, side=1):
    a.boxplot(data, positions=pos, widths=w, whiskerprops={"color": c}, capprops={"color": c},
              boxprops={"facecolor": c, "edgecolor": c}, **BOXKW)
    for p_, d in zip(pos, data):  # write the median next to each box
        md_ = np.median(d)
        a.text(p_ + side * (w / 2 + 0.03), md_, f"{md_:.1f}" if md_ < 100 else f"{md_:.0f}",
               ha="left" if side > 0 else "right", va="center",
               fontsize=8.5, color=INK, fontweight="bold", zorder=5,
               bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9})

fig, ax = plt.subplots(2, 1, figsize=(6.0, 8.9), gridspec_kw={"hspace": 0.5, "height_ratios": [1, 1.15]})
# a -- does the measured difference depend on how many apartments were monitored?
pos = np.arange(4)
cats = list(s["grp"].cat.categories)
for off, col, lab, c in ((-0.19, "raw", "Warmest − coolest of all monitored apartments\n(unadjusted range)", ORANGE),
                         (0.19, "three", "Warmest − coolest of three random apartments\n(expected range, mean of 500 draws)", BLUE)):
    boxes(ax[0], [s.loc[s["grp"] == g, col].to_numpy() for g in cats], pos + off, c, 0.32, side=np.sign(off))
    ax[0].plot([], [], color=c, lw=7, label=lab)
ax[0].set_xticks(pos, [f"{g}\n({(s['grp'] == g).sum()} bldgs)" for g in cats], fontsize=9)
ax[0].set_xlabel("Number of monitored apartments in the building", fontsize=FS)
ax[0].set_ylabel("Difference in summer mean\nindoor air temperature (°C)", fontsize=FS)
ax[0].set_title("a  Temperature difference between apartments\n    of the same building", fontsize=FT)
ax[0].set_ylim(0, 7.6)
ax[0].legend(loc="upper left", fontsize=8.3, handlelength=1.2, labelspacing=0.7)
# b -- is the difference larger during the heatwaves?
mb = m.loc[b3, ["bld_spread_mean", "bld_hw_spread_mean"]].dropna()
above = (mb["bld_hw_spread_mean"] > mb["bld_spread_mean"]).mean() * 100
lim = [0, float(np.ceil(mb.max().max()))]
ax[1].fill_between(lim, lim, [lim[1]] * 2, color=ORANGE, alpha=0.08, lw=0)
ax[1].scatter(mb["bld_spread_mean"], mb["bld_hw_spread_mean"], s=15, color=BLUE, alpha=0.55, edgecolor="none")
ax[1].plot(lim, lim, color=INK2, lw=1.1, ls="--")
ax[1].text(0.3, lim[1] - 0.3, f"Larger difference\nduring heatwaves:\n{above:.0f}% of buildings", color="#9e3a10",
           fontsize=9, ha="left", va="top", fontweight="bold")
ax[1].text(lim[1] - 0.3, 0.3, f"Smaller difference\nduring heatwaves:\n{100 - above:.0f}% of buildings", color=INK2,
           fontsize=9, ha="right", va="bottom")
ax[1].text(lim[1] * 0.74, lim[1] * 0.74 - 0.35, "1:1 line (equal difference)", color=INK2, fontsize=8.5,
           rotation=33, ha="center", va="top")
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("Warmest − coolest apartment, whole summer (°C)", fontsize=FS)
ax[1].set_ylabel("Warmest − coolest apartment,\n22 heatwave days (°C)", fontsize=FS)
ax[1].set_title(f"b  Temperature difference between apartments: heatwaves\n    vs whole summer (one point = one building, n = {len(mb)})", fontsize=FT)
for a in ax:
    a.tick_params(labelsize=9)
fig.text(0.02, 0.005, "Panel a: boxes show the middle 50% of buildings; white line and number: median; whiskers: 1.5 × interquartile range.",
         fontsize=8.3, color=INK2, ha="left", va="bottom")
fig.savefig(os.path.join(PLOTS, "fig3_within_building_variation.png"), dpi=450)
plt.close(fig)
print("saved fig3_within_building_variation.png")

# ---------------------------------------------------------------- Fig 5
icc = {}
for col, lab in (("temp_mean", "Summer mean\ntemperature"), ("temp_max", "Summer maximum\ntemperature")):
    dd = apt[[col, "building_id"]].dropna()
    f = smf.mixedlm(f"{col} ~ 1", dd, groups=dd["building_id"]).fit(reml=True)
    bv, wv = float(f.cov_re.iloc[0, 0]), float(f.scale)
    icc[lab] = bv / (bv + wv)
r2 = pd.read_csv(os.path.join(EXPORTS, "step2_r2.csv")).set_index("col")
t4 = pd.read_csv(os.path.join(EXPORTS, "table4_r2.csv")).set_index("measure")
items = [("Building mean temperature", r2.loc["bld_mean_tempmean", "R2_full"], r2.loc["bld_mean_tempmean", "R2_RF"]),
         ("Share of overheated apartments", r2.loc["frac_over", "R2_full"], r2.loc["frac_over", "R2_RF"]),
         ("Building mean of maximum temperatures", r2.loc["bld_mean_peak_temp", "R2_full"], r2.loc["bld_mean_peak_temp", "R2_RF"]),
         ("Mean degree-hours above 27 °C", r2.loc["bld_mean_dh27", "R2_full"], r2.loc["bld_mean_dh27", "R2_RF"]),
         ("Unadjusted range", t4.loc["spread_raw", "R2_linear"], t4.loc["spread_raw", "R2_RF"]),
         ("Range of three apartments", t4.loc["spread_fixed3", "R2_linear"], t4.loc["spread_fixed3", "R2_RF"]),
         ("Standard deviation", t4.loc["sd_temp", "R2_linear"], t4.loc["sd_temp", "R2_RF"]),
         ("Degree-hours of warmest apartment", r2.loc["bld_peak_dh27", "R2_full"], r2.loc["bld_peak_dh27", "R2_RF"])]
from decimal import Decimal, ROUND_HALF_UP


def r2_label(v):
    """Two decimals, halves rounded up (0.235 -> 0.24, as in the text), true minus sign."""
    t = str(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return t.replace("-", "−")


# vertical layout: (a) above (b), labels on the left, 450 dpi
fig, ax = plt.subplots(2, 1, figsize=(6.4, 7.6), gridspec_kw={"hspace": 0.55, "height_ratios": [0.42, 1]})
y = np.arange(len(icc))[::-1]
for yi, (lab, v_) in zip(y, icc.items()):
    ax[0].barh(yi, v_ * 100, color=BLUE, height=0.55, edgecolor="white", linewidth=2)
    ax[0].barh(yi, 100 - v_ * 100, left=v_ * 100, color="#c9c8c2", height=0.55, edgecolor="white", linewidth=2)
    ax[0].text(v_ * 50, yi, f"{v_:.0%}", color="white", ha="center", va="center", fontsize=10, fontweight="bold")
    ax[0].text(v_ * 100 + (100 - v_ * 100) / 2, yi, f"{1 - v_:.0%}", color=INK, ha="center", va="center", fontsize=10, fontweight="bold")
ax[0].set_yticks(y, [k.replace(chr(10), " ") for k in icc], fontsize=9.5)
ax[0].set_xlim(0, 100)
ax[0].set_xlabel("Share of variance of apartment temperatures (%)", fontsize=9.5)
ax[0].tick_params(axis="x", labelsize=9)
ax[0].grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color=BLUE, label="Between buildings (ICC)"), Patch(color="#c9c8c2", label="Within buildings")],
             loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
ax[0].set_title("a  Variance between and within buildings", fontsize=10.5, pad=26)
yy = np.arange(len(items))[::-1].astype(float)
yy[4:] -= 0.7
h = 0.38
ax[1].barh(yy + h / 2, [i[1] for i in items], height=h, color=BLUE, label="Linear regression")
ax[1].barh(yy - h / 2, [i[2] for i in items], height=h, color=ORANGE, label="Random forest")
for yi, (_, lin, rf) in zip(yy, items):
    for val, off in ((lin, h / 2), (rf, -h / 2)):
        ax[1].text(val + (0.006 if val >= 0 else -0.006), yi + off, r2_label(val), va="center",
                   ha="left" if val >= 0 else "right", fontsize=8, color=INK)
ax[1].axvline(0, color=INK, lw=0.8)
ax[1].set_yticks(yy, [i[0] for i in items], fontsize=9.5)
ax[1].set_xlabel("Cross-validated R²", fontsize=9.5)
ax[1].tick_params(axis="x", labelsize=9)
ax[1].grid(axis="y", visible=False)
ax[1].text(-0.235, yy[0] + 0.62, "Building-level conditions", fontsize=9.5, color=INK2, ha="left", style="italic", fontweight="bold")
ax[1].text(-0.235, yy[4] + 0.62, "Within-building variation", fontsize=9.5, color=INK2, ha="left", style="italic", fontweight="bold")
ax[1].set_xlim(-0.24, 0.3)
ax[1].legend(loc="lower right", ncol=1, fontsize=9)
ax[1].set_title("b  Predictability from public building and microclimate data", fontsize=10.5, pad=18)
fig.savefig(os.path.join(PLOTS, "fig4_variance_and_predictability.png"), dpi=450)
plt.close(fig)
print("saved fig4_variance_and_predictability.png")

# ---------------------------------------------------------------- Fig 6
oof = pd.read_csv(os.path.join(EXPORTS, "oof_point_scores.csv"))
TLAB = {"A": "Target A: composition risk", "B": "Target B: severity risk"}
fig, ax = plt.subplots(2, 1, figsize=(6.0, 9.4), gridspec_kw={"hspace": 0.5, "height_ratios": [1.1, 1]})
cap30 = {}
for key, c, lw, ls in (("A", BLUE, 2.2, "-"), ("B", ORANGE, 2.2, (0, (5, 2)))):
    o = oof[oof["target"] == key]
    yh = (o["tier"] == "High").to_numpy()
    sc = o["score"].to_numpy()
    curves = []
    for _ in range(200):                                  # ties broken at random, averaged
        order = np.lexsort((rng.random(len(sc)), -sc))
        curves.append(np.concatenate([[0], np.cumsum(yh[order]) / yh.sum()]))
    cur = np.mean(curves, axis=0)
    xs = np.arange(len(cur)) / (len(cur) - 1)
    cap30[key] = yh[o["priority"].to_numpy()].sum() / yh.sum() * 100   # same Priority group as in the text
    ax[0].plot(xs * 100, cur * 100, color=c, lw=lw, ls=ls, label=TLAB[key])
ax[0].plot([0, 100], [0, 100], color=INK2, lw=1, ls=":", label="Random selection")
ax[0].axvline(30, color=INK2, lw=1, ls="--", zorder=1)
ax[0].text(31.5, 97, "Priority group:\n30% of buildings\nwith the highest scores", fontsize=8.5, color=INK2, ha="left", va="top")
ax[0].annotate(f"{cap30['A']:.0f}% (A) and {cap30['B']:.0f}% (B)\nof high-risk\nbuildings captured\n(30% at random)",
               xy=(30, cap30["A"]), xytext=(2, 92), fontsize=8.5, color=INK, ha="left", va="top",
               arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9})
ax[0].set_xlim(0, 100); ax[0].set_ylim(0, 100)
ax[0].set_xlabel("Share of buildings selected for assessment, highest scores first (%)", fontsize=FS)
ax[0].set_ylabel("Share of all high-risk buildings\ncaptured (%)", fontsize=FS)
ax[0].set_title("a  How many high-risk buildings are found when buildings\n    are assessed in order of their point score", fontsize=FT)
ax[0].legend(loc="lower right", fontsize=8.5)
for i, key in enumerate(("A", "B")):
    o = oof[oof["target"] == key]
    yh = o["tier"] == "High"
    pri, std = yh[o["priority"]].mean() * 100, yh[~o["priority"]].mean() * 100
    base = yh.mean() * 100
    for off, val, c, lab in ((-0.2, pri, ORANGE, "Priority group (30% with the highest scores)"),
                             (0.2, std, "#c9c8c2", "Standard group (other 70%)")):
        ax[1].bar(i + off, val, width=0.36, color=c, label=lab if i == 0 else None)
        ax[1].text(i + off, val - 2.5, f"{val:.0f}%", ha="center", va="top", fontsize=10,
                   color="white" if c == ORANGE else INK, fontweight="bold")
    ax[1].hlines(base, i - 0.42, i + 0.42, color=INK, lw=1.1, ls="--", label="All buildings" if i == 0 else None)
    ax[1].text(i + 0.43, base, f"{base:.0f}%", ha="left", va="center", fontsize=8.5, color=INK)
ax[1].set_xticks([0, 1], ["Target A: composition risk\n(share of overheated apartments)",
                          "Target B: severity risk\n(heatwave temperatures)"], fontsize=9)
ax[1].set_ylim(0, 112); ax[1].set_yticks(range(0, 101, 20))
ax[1].set_ylabel("High-risk buildings\nin the group (%)", fontsize=FS)
ax[1].grid(axis="x", visible=False)
ax[1].set_title("b  Share of high-risk buildings in the Priority\n    and Standard groups", fontsize=FT)
ax[1].legend(loc="upper center", fontsize=8.5, ncol=1, bbox_to_anchor=(0.5, 1.0))
for a_ in ax:
    a_.tick_params(labelsize=9)
fig.savefig(os.path.join(PLOTS, "fig5_prioritisation.png"), dpi=450)
plt.close(fig)
print("saved fig5_prioritisation.png", {k: round(v, 1) for k, v in cap30.items()})
print("medians: range raw %.2f three %.2f | summer %.2f heatwave %.2f | ICC %s" % (
    s["raw"].median(), s["three"].median(), mb["bld_spread_mean"].median(), mb["bld_hw_spread_mean"].median(),
    {k.replace(chr(10), " "): round(v, 3) for k, v in icc.items()}))
