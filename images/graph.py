import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# =====================================================
# Smooth continuous function
# =====================================================
x = np.linspace(0, 10, 600)
y = 0.35*(x-5)**2*np.sin(x/2)/3 + np.sin(x/1.6)

# =====================================================
# Figure
# =====================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

# Common settings
for ax in [ax1, ax2]:
    ax.set_xlim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Hide all default borders
    for spine in ax.spines.values():
        spine.set_visible(False)

ymin = np.min(y)
ymax = np.max(y)

# Position of custom x-axis
axis_y = ymin - 0.22

# Leave plenty of space below for captions
bottom_margin = 0.90

for ax in [ax1, ax2]:
    ax.set_ylim(axis_y-bottom_margin, ymax+0.15)

# =====================================================
# LEFT : Continuous Function
# =====================================================

ax1.plot(x, y, color="#0B57D0", lw=2.3)

# Custom axes
ax1.annotate("", xy=(10, axis_y), xytext=(0, axis_y),
             arrowprops=dict(arrowstyle="->", lw=1.7))

ax1.annotate("", xy=(0, ymax+0.1), xytext=(0, axis_y),
             arrowprops=dict(arrowstyle="->", lw=1.7))

ax1.text(10.15, axis_y-0.02, "x", fontsize=12)
ax1.text(-0.45, ymax+0.08, "u(x)", fontsize=12)

ax1.set_title(
    "Continuous Function",
    fontsize=16,
    color="#0B3D91",
    weight="bold"
)

# Caption (well below x-axis)
ax1.text(
    5,
    axis_y-0.62,
    "Infinitely many values",
    ha="center",
    va="top",
    fontsize=11
)

# =====================================================
# RIGHT : Discrete Samples
# =====================================================

ax2.plot(
    x,
    y,
    "--",
    color="#0B57D0",
    lw=2
)

xs = np.linspace(1,10,8)
ys = 0.35*(xs-5)**2*np.sin(xs/2)/3 + np.sin(xs/1.6)

# Vertical dashed sampling lines
for xx,yy in zip(xs,ys):
    ax2.plot(
        [xx,xx],
        [axis_y,yy],
        "--",
        color="lightgray",
        lw=0.9
    )

# Sample points
ax2.scatter(
    xs,
    ys,
    s=32,
    color="#0B57D0",
    zorder=5
)

# Custom axes
ax2.annotate("", xy=(10, axis_y), xytext=(0, axis_y),
             arrowprops=dict(arrowstyle="->", lw=1.7))

ax2.annotate("", xy=(0, ymax+0.1), xytext=(0, axis_y),
             arrowprops=dict(arrowstyle="->", lw=1.7))

# Tick labels
labels = [
    r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$",
    r"$x_5$", r"$x_6$", "...", r"$x_n$"
]

for xx,lab in zip(xs,labels):
    ax2.text(
        xx,
        axis_y-0.09,
        lab,
        ha="center",
        va="top",
        fontsize=11
    )

ax2.text(10.15, axis_y-0.02, "x", fontsize=12)
ax2.text(-0.45, ymax+0.08, "u(x)", fontsize=12)

ax2.set_title(
    "Discrete Samples",
    fontsize=16,
    color="#0B3D91",
    weight="bold"
)

# Caption (far enough below tick labels)
ax2.text(
    5,
    axis_y-0.62,
    "Finite set of samples",
    ha="center",
    va="top",
    fontsize=11
)

# =====================================================
# Middle arrow
# =====================================================

arrow_ax = fig.add_axes([0.455,0.33,0.09,0.28])
arrow_ax.axis("off")

arrow = FancyArrowPatch(
    (0.05,0.5),
    (0.95,0.5),
    arrowstyle="simple",
    mutation_scale=38,
    color="gray",
    alpha=0.55
)

arrow_ax.add_patch(arrow)

arrow_ax.text(
    0.5,
    0.78,
    "Discretization",
    ha="center",
    fontsize=11
)

plt.subplots_adjust(wspace=0.38)

plt.savefig(
    "discretization_publication.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()