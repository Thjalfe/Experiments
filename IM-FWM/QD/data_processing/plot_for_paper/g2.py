import matplotlib.pyplot as plt
from typing import cast, Callable
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from funcs.g2_processing import (
    downsample_by_n,
    normalize_data,
    find_bunching_ratio_from_ref,
    simple_weighting_around_dip,
    exp_decay_bunching,
    cw_g2_bunching,
    # plot_g2,
)
import numpy as np


plt.style.use("custom")

plt.style.use("custom")
figsize = (11, 8)
plt.rcParams["figure.figsize"] = figsize
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{upgreek}\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{courier}\usepackage{helvet}"
)
save_figs = True
# plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["axes.labelsize"] = 48
plt.rcParams["xtick.labelsize"] = 36
plt.rcParams["ytick.labelsize"] = 36
plt.rcParams["legend.title_fontsize"] = 28

paper_dir = "/home/thjalfe/Documents/PhD/Projects/papers/FC_QD/figs"
save_figs = True
ref_file_name = (
    "../../data/TT_g2/ref/Bidir_hist_2024-09-25_CW_p-shell_971p22nm_13p3mW.txt"
)
idler_file_name = "../../data/TT_g2/g2_converted_idler/Bidirectional_histogram_2024-09-27_090258_19.txt"
ref_data = np.loadtxt(ref_file_name, skiprows=1)
idler_data = np.loadtxt(idler_file_name, skiprows=1)
ref_data[:, 0] = ref_data[:, 0] * 1e-3  # ns
idler_data[:, 0] = idler_data[:, 0] * 1e-3  # ns
ref_idx_min = np.argmin(np.abs(ref_data[:, 1]))
idler_idx_min = np.argmin(np.abs(idler_data[:, 1]))
# offset such that min is centered at 0
ref_data[:, 0] = ref_data[:, 0] - ref_data[ref_idx_min, 0]
idler_data[:, 0] = idler_data[:, 0] - idler_data[idler_idx_min, 0]


time_window = 20
normalized_ref_data, ref_norm_factor_full = normalize_data(ref_data, 100)
ref_bunching_diff = find_bunching_ratio_from_ref(ref_data, time_window)
normalized_idler_data, idler_norm_factor = normalize_data(idler_data, 4)

normalized_idler_data[:, 1] = normalized_idler_data[:, 1] / ref_bunching_diff


sample_factor_ref = 8
sample_factor = 2


weight = 0.1
# weight = 1
center_range = 750
# center_range = 1050
# center_range = 2000
normalized_ref_data = downsample_by_n(normalized_ref_data, sample_factor_ref)
weights = simple_weighting_around_dip(normalized_ref_data, center_range, weight)
(
    fit_result_ref,
    residuals_ref,
    jacobian_ref,
    cov_matrix_ref,
    std_ref,
    norm_val_ref,
) = cw_g2_bunching(
    normalized_ref_data[:, 0], normalized_ref_data[:, 1], weights=weights
)
# remove_idx_ref = np.logical_or(
#     normalized_ref_data[:, 0] < -20, normalized_ref_data[:, 0] > 20
# )
# normalized_ref_data = normalized_ref_data[~remove_idx_ref]
# weights = weights[~remove_idx_ref]
new_x_ax_ref = np.linspace(
    np.min(normalized_ref_data[:, 0]),
    np.max(normalized_ref_data[:, 0]),
    len(normalized_ref_data[:, 0]) * 100,
)


def plot_g2(
    time: np.ndarray,
    time_fit: np.ndarray,
    hist_norm: np.ndarray,
    fit_result: np.ndarray,
    ci: np.ndarray,
    exp_fun: Callable,
    markerstyle: str = "o",
    markersize: int = 2,
    plot_confidence: bool = True,
    plotting_style="markers",
    plot_every_n=1,
    bar_width=0.1,
) -> tuple[Figure, Axes]:
    import seaborn as sns

    bar_color = sns.color_palette("pastel")[0]  # light gray or pastel
    fit_line_color = sns.color_palette("bright")[3]  # bright yellow
    fig, ax = plt.subplots()
    fig = cast(Figure, fig)
    ax = cast(Axes, ax)
    if plotting_style == "markers":
        ax.plot(
            time,
            hist_norm,
            markerstyle,
            markersize=markersize,
            linewidth=2,
            label="Raw data",
        )
    elif plotting_style == "bars":
        ax.bar(
            time[::plot_every_n],
            hist_norm[::plot_every_n],
            width=bar_width,
            label="Raw data",
            color=bar_color,
        )
        ax.grid(True)
    ax.plot(
        time_fit,
        exp_fun(time_fit, *fit_result),
        linewidth=2,
        label="Fit",
        # color="C3",
        color=fit_line_color,
        linestyle="--",
    )
    if plot_confidence:
        lower_bound = exp_fun(time_fit, *(fit_result - ci))
        upper_bound = exp_fun(time_fit, *(fit_result + ci))
        ax.fill_between(
            time_fit, lower_bound, upper_bound, color="gray", alpha=0.3, label="95% CI"
        )
    ax.set_xlabel(r"$\tau_\mathrm{delay}$ [ns]")
    ax.set_ylabel(r"$g^{(2)}(\tau)$")
    return fig, ax


fig_ref, ax_ref = plot_g2(
    normalized_ref_data[:, 0],
    new_x_ax_ref,
    normalized_ref_data[:, 1],
    fit_result_ref,
    std_ref,
    exp_decay_bunching,
    plot_confidence=False,
    plotting_style="bars",
)
ax_ref.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=4))
ax_ref.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax_ref.set_xlim(-18, 18)
ax_ref.set_ylim(0, 2)
ax_ref.set_xlabel("")
ax_ref.set_ylabel("")
fig_ref.set_size_inches(11, 4)
if save_figs:
    fig_ref.savefig(f"{paper_dir}/ref_g2.pdf", bbox_inches="tight")
    ax_ref.set_xlim(-400, 400)

fig_ref, ax_ref = plot_g2(
    normalized_ref_data[:, 0],
    new_x_ax_ref,
    normalized_ref_data[:, 1],
    fit_result_ref,
    std_ref,
    exp_decay_bunching,
    plot_confidence=False,
    plotting_style="bars",
    plot_every_n=7,
    bar_width=0.7,
)
ax_ref.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax_ref.set_xlim(-290, 290)
ax_ref.set_ylim(0, 2)
ax_ref.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
rect = Rectangle(
    (-18, 0.01),
    36,
    1.97,  # (x, y), width, height
    linewidth=2,
    edgecolor="k",
    facecolor="none",
    linestyle="--",
    alpha=1,
)
ax_ref.add_patch(rect)
if save_figs:
    fig_ref.set_size_inches(11, 8)
    fig_ref.savefig(f"{paper_dir}/ref_g2_full_window.pdf", bbox_inches="tight")

center_range_idler = 147  # 147
normalized_idler_data = downsample_by_n(normalized_idler_data, sample_factor)
weights = simple_weighting_around_dip(normalized_idler_data, center_range_idler, weight)
(
    fit_result_idler,
    residuals_idler,
    jacobian_idler,
    cov_matrix_idler,
    ci_idler,
    norm_val_idler,
) = cw_g2_bunching(
    normalized_idler_data[:, 0], normalized_idler_data[:, 1], weights=weights
)
new_x_ax_idler = np.linspace(
    np.min(normalized_idler_data[:, 0]),
    np.max(normalized_idler_data[:, 0]),
    len(normalized_idler_data[:, 0]) * 100,
)
fig_idler, ax_idler = plot_g2(
    normalized_idler_data[:, 0],
    new_x_ax_idler,
    normalized_idler_data[:, 1],
    fit_result_idler,
    ci_idler,
    exp_decay_bunching,
    markerstyle="o",
    markersize=4,
    plot_confidence=False,
    plotting_style="bars",
)
ax_idler.set_xlim(-18, 18)
ax_idler.set_ylim(0, 2)
ax_idler.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax_idler.set_ylabel("")
ax_idler.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=4))
ax_idler.set_ylabel("")
fig_idler.set_size_inches(11, 4)
if save_figs:
    fig_idler.savefig(f"{paper_dir}/idler_g2.pdf", bbox_inches="tight")

plt.close()
print(
    f"g2 ref: {1-fit_result_ref[0]+fit_result_ref[2]:.2f}+-{np.sqrt(std_ref[0]**2+std_ref[2]**2):.2f}"
)
print(
    f"g2 idler: {1-fit_result_idler[0]+fit_result_idler[2]:.3f}+-{np.sqrt(ci_idler[0]**2+ci_idler[2]**2):.3f}"
)


# |%%--%%| <BQ2l6UxuvO|c408qMx6fR>
def plot_g2_subplot(
    time_ref: np.ndarray,
    time_ref_fit: np.ndarray,
    hist_ref_norm: np.ndarray,
    fit_ref_result: np.ndarray,
    time_idler: np.ndarray,
    time_idler_fit: np.ndarray,
    hist_idler_norm: np.ndarray,
    fit_idler_result: np.ndarray,
    exp_fun: Callable,
) -> tuple[Figure, Axes, Axes, Axes]:
    import seaborn as sns

    bar_color = sns.color_palette("pastel")[0]  # light gray or pastel
    fit_line_color = sns.color_palette("bright")[3]  # bright yellow
    fig = plt.figure(figsize=(22, 8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs.update(wspace=0.22, hspace=0.3)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.bar(
        time_ref[::7], hist_ref_norm[::7], width=0.7, label="Raw data", color=bar_color
    )
    # ax1.grid(False)
    ax1.plot(
        time_ref_fit,
        exp_fun(time_ref_fit, *fit_ref_result),
        linewidth=2,
        label="Fit",
        # color="C3",
        color=fit_line_color,
        linestyle="--",
    )
    ax1.set_xlim(-390, 390)
    # ax1.set_xlim(-290, -200)
    ax1.set_xlabel(r"$\tau_\mathrm{delay}$ [ns]")
    ax1.set_ylabel(r"$g^{(2)}(\tau)$")
    ax1.set_ylim(0, 2)
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    rect = Rectangle(
        (-18, 0.01),
        36,
        1.97,  # (x, y), width, height
        linewidth=2,
        edgecolor="k",
        facecolor="none",
        linestyle="--",
        alpha=1,
    )
    ax1.add_patch(rect)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(time_ref, hist_ref_norm, width=0.1, label="Raw data", color=bar_color)
    # ax2.grid(False)
    ax2.plot(
        time_ref_fit,
        exp_fun(time_ref_fit, *fit_ref_result),
        linewidth=2,
        color=fit_line_color,
        linestyle="--",
    )
    ax2.set_xlim(-18, 18)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=4))
    ax2.set_ylim(0, 2)
    ax2.set_xticks([])
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(time_idler, hist_idler_norm, width=0.1, label="Raw data", color=bar_color)
    # ax3.grid(False)
    ax3.plot(
        time_idler_fit,
        exp_fun(time_idler_fit, *fit_idler_result),
        linewidth=2,
        color=fit_line_color,
        linestyle="--",
    )
    ax3.set_xlim(-18, 18)
    ax3.set_xlabel(r"$\tau_\mathrm{delay}$ [ns]")
    ax3.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=4))
    ax3.set_ylim(0, 2)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    textbox_names = ["Signal", "Signal", "Converted signal"]
    xy_coord_arr = [(0.15, 0.95), (0.215, 0.9), (0.4, 0.9)]
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.annotate(
            textbox_names[i],
            xy=xy_coord_arr[i],
            xycoords="axes fraction",
            fontsize=28,
            ha="right",
            va="top",
        )
    return fig, ax1, ax2, ax3


fig_both, ax_both_1, ax_both_2, ax_both_3 = plot_g2_subplot(
    normalized_ref_data[:, 0],
    new_x_ax_ref,
    normalized_ref_data[:, 1],
    fit_result_ref,
    normalized_idler_data[:, 0],
    new_x_ax_idler,
    normalized_idler_data[:, 1],
    fit_result_idler,
    exp_decay_bunching,
)
ax_both_1.text(
    -0.18,
    1.06,
    r"\textbf{a}",
    transform=ax_both_1.transAxes,
    fontsize=40,
    va="top",
    ha="left",
)
ax_both_2.text(
    -0.17,
    1.12,
    r"\textbf{b}",
    transform=ax_both_2.transAxes,
    fontsize=40,
    va="top",
    ha="left",
)
ax_both_3.text(
    -0.17,
    1.12,
    r"\textbf{c}",
    transform=ax_both_3.transAxes,
    fontsize=40,
    va="top",
    ha="left",
)
if save_figs:
    fig_both.savefig(f"{paper_dir}/g2_fig_combined.pdf", bbox_inches="tight")
# plt.show()
