"""
Paper figure: study workflow. Saves plots/fig_workflow.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.environ.get("OVERHEATING_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))), "plots", "fig_workflow.png")
INK = "#1E2329"
DATA, STEP, OUTC, LINE = "#E8EEF3", "#1F5F8B", "#F4D9C6", "#8A949E"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(6.7, 4.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.6)
ax.axis("off")


def box(x, y, w, h, title, body, fill, col=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fill, ec="none"))
    ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="top", fontsize=7.4, fontweight="bold", color=col, linespacing=1.15)
    if body:
        n = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 0.2 - 0.3 * n, body, ha="center", va="top", fontsize=6.5, color=col, linespacing=1.25)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, color=LINE, lw=1))


box(0.2, 5.0, 3.0, 1.45, "Indoor temperatures", "Hourly, 15 May–31 Aug 2021\n5,978 apartments, 568 buildings", DATA)
box(3.47, 5.0, 3.0, 1.45, "Outdoor weather", "Helsinki-Vantaa station\nheatwave periods", DATA)
box(6.74, 5.0, 3.0, 1.45, "Public building and\nmicroclimate data", "register, land cover,\nroads, coastline", DATA)
box(0.2, 3.95, 6.27, 0.8, "Overheating indicators", "degree-hours above 27 and 30 °C, mean and maximum temperature", DATA)
arrow(1.7, 5.0, 1.7, 4.77)
arrow(4.97, 5.0, 4.97, 4.77)

steps = [("1  Variation\nwithin buildings", "range of apartments,\ncoolest apartment,\nbuilding age"),
         ("2  What public\ndata predict", "ICC,\ncross-validated R²"),
         ("3  Building risk\ndefinitions", "composition and\nseverity,\nk-means clustering"),
         ("4  Prioritisation\nmethod", "point score,\nbenchmarked against\ntree and ML models")]
xs = [0.2, 2.67, 5.14, 7.61]
for x, (t, b) in zip(xs, steps):
    box(x, 1.75, 2.2, 1.85, t, b, STEP, "white")
for x in xs[:3]:
    arrow(x + 1.1, 3.95, x + 1.1, 3.62)
arrow(8.24, 5.0, 8.71, 3.62)
arrow(xs[2] + 2.2, 2.3, xs[3] - 0.02, 2.3)

box(0.2, 0.15, 4.5, 1.1, "Apartment level", "hottest apartments not predictable from\npublic data; next step: simulation", OUTC)
box(5.14, 0.15, 4.67, 1.1, "Building level", "ranked list of buildings\nfor renovation assessment", OUTC)
arrow(1.3, 1.75, 1.3, 1.27)
arrow(8.71, 1.75, 8.71, 1.27)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
print(os.path.abspath(OUT))
