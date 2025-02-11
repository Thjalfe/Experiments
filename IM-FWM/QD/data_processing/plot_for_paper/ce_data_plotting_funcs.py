import numpy as np
from matplotlib.lines import Line2D
from functools import partial
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import cast


def thz_to_wls(thz, thz_sep, idler_wls):
    poly_coeffs = np.polyfit(thz_sep, idler_wls, 1)
    poly = np.poly1d(poly_coeffs)
    return poly(thz)


def wls_to_thz(wls, thz_sep, idler_wls):
    poly_coeffs = np.polyfit(idler_wls, thz_sep, 1)
    poly = np.poly1d(poly_coeffs)
    return poly(wls)


def return_red_and_blue_idxs(thz_sep: np.ndarray):
    red_idxs = np.where(thz_sep < 0)[0]
    blue_idxs = np.where(thz_sep > 0)[0]
    return red_idxs, blue_idxs


def plot_ce_vs_detuning_all_dc(
    dc_arr, ce_pol_opt_peak_mean, ce_pol_opt_peak_std, pump_sep_thz
):
    dc_percentage = 100 * dc_arr
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    fig = cast(Figure, fig)
    for dc_idx, dc in enumerate(dc_arr):
        dc_per = dc_percentage[dc_idx]
        ax.errorbar(
            pump_sep_thz,
            ce_pol_opt_peak_mean[:, dc_idx],
            yerr=ce_pol_opt_peak_std[:, dc_idx],
            label=rf"DC={dc_per:.1f}\%",
            fmt="-o",
        )
    ax.set_xlabel("Pump separation (THz)")
    ax.set_ylabel("CE peak (pol opt)")
    ax.legend()
    return fig, ax


def plot_ce_diff_dc_rel_to_cw(dc_arr, ce_pol_opt_peak_mean, pump_sep_thz):
    ce_diff_between_dc_rel_to_cw = np.array(
        [
            ce_pol_opt_peak_mean[:, dc_idx] - ce_pol_opt_peak_mean[:, -1]
            for dc_idx in range(len(dc_arr))
        ]
    )
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    fig = cast(Figure, fig)
    for pump_sep_idx in range(len(pump_sep_thz)):
        ax.plot(
            -10 * np.log10(dc_arr),
            ce_diff_between_dc_rel_to_cw[:, pump_sep_idx],
            "-o",
            label=f"{pump_sep_thz[pump_sep_idx]:.2f} THz",
        )
        ax.plot(-10 * np.log10(dc_arr), -20 * np.log10(dc_arr), "--", color="black")
    ax.set_xlabel("1 / DC")
    ax.set_ylabel("CE peak difference rel to CW")
    ax.legend(title="Pump separation [THz]")
    return fig, ax


def plot_p1_vs_p2_phasematch(p1_wl, p2_wl):
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    fig = cast(Figure, fig)
    ax.plot(p1_wl, p2_wl, "-o")
    return fig, ax


def plot_ce_vs_abs_detuning_w_qd(
    qd_thz_sep: np.ndarray,
    qd_ces: np.ndarray,
    qd_stds: np.ndarray,
    classical_pump_sep_thz: np.ndarray,
    ce_classical_mean: np.ndarray,
    colors: list,
    qd_red_idxs_to_ignore: np.ndarray = np.array([]),
    qd_blue_idxs_to_ignore: np.ndarray = np.array([]),
):
    classical_red_idxs, classical_blue_idxs = return_red_and_blue_idxs(
        classical_pump_sep_thz
    )
    qd_red_idxs, qd_blue_idxs = return_red_and_blue_idxs(qd_thz_sep)
    qd_red_idxs = np.setdiff1d(qd_red_idxs, qd_red_idxs_to_ignore + qd_red_idxs[0])
    qd_blue_idxs = np.setdiff1d(qd_blue_idxs, qd_blue_idxs_to_ignore + qd_blue_idxs[0])
    qd_blue_ces = qd_ces[qd_blue_idxs]
    qd_blue_stds = qd_stds[qd_blue_idxs]
    qd_red_ces = qd_ces[qd_red_idxs]
    qd_red_stds = qd_stds[qd_red_idxs]
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    fig = cast(Figure, fig)

    ax.plot(
        np.abs(classical_pump_sep_thz[classical_blue_idxs]),
        ce_classical_mean[classical_blue_idxs, 0],
        "x",
        color=colors[0],
    )
    ax.plot(
        np.abs(classical_pump_sep_thz[classical_red_idxs]),
        ce_classical_mean[classical_red_idxs, 0],
        "x",
        color=colors[3],
    )
    ax.errorbar(
        np.abs(qd_thz_sep[qd_blue_idxs]),
        qd_blue_ces,
        yerr=qd_blue_stds,
        fmt="o",
        markersize=8,
        capsize=8,
    )
    ax.errorbar(
        np.abs(qd_thz_sep[qd_red_idxs]),
        qd_red_ces,
        yerr=qd_red_stds,
        fmt="o",
        markersize=8,
        capsize=8,
        color=colors[3],
    )

    ax.plot([], [], "x", color="k", label="Classical")
    ax.errorbar([], [], color="k", fmt="o", label="Quantum")
    ax.set_xlabel(r"$\nu_\mathrm{i}-\nu_\mathrm{s}$ [THz]")
    ax.set_ylabel(r"$\eta_{\mathrm{peak}}$ [\%]")
    ax.legend()
    return fig, ax


def plot_ce_vs_detuning_freq_bottom_ax(
    qd_thz_sep: np.ndarray,
    qd_ces: np.ndarray,
    qd_stds: np.ndarray,
    classical_pump_sep_thz: np.ndarray,
    ce_classical_mean: np.ndarray,
    ce_classical_std: np.ndarray,
    idler_wls: np.ndarray,
    colors: list,
    qd_red_idxs_to_ignore: np.ndarray = np.array([]),
    qd_blue_idxs_to_ignore: np.ndarray = np.array([]),
    show_classical_error_bars: bool = True,
    extra_data_idler_wls: np.ndarray | None = None,
    extra_data_ce_mean: np.ndarray | None = None,
):
    classical_red_idxs, classical_blue_idxs = return_red_and_blue_idxs(
        classical_pump_sep_thz
    )
    qd_red_idxs, qd_blue_idxs = return_red_and_blue_idxs(qd_thz_sep)
    qd_red_idxs = np.setdiff1d(qd_red_idxs, qd_red_idxs_to_ignore + qd_red_idxs[0])
    qd_blue_idxs = np.setdiff1d(qd_blue_idxs, qd_blue_idxs_to_ignore + qd_blue_idxs[0])
    qd_blue_ces = qd_ces[qd_blue_idxs]
    qd_blue_stds = qd_stds[qd_blue_idxs]
    qd_red_ces = qd_ces[qd_red_idxs]
    qd_red_stds = qd_stds[qd_red_idxs]
    if show_classical_error_bars:
        capsize = 8
    else:
        capsize = 0
        ce_classical_std = np.zeros_like(ce_classical_std)
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    fig = cast(Figure, fig)

    ax.errorbar(
        classical_pump_sep_thz[classical_red_idxs],
        np.squeeze(ce_classical_mean[classical_red_idxs, 0]),
        yerr=np.squeeze(ce_classical_std[classical_red_idxs, 0]),
        capsize=capsize,
        fmt="-o",
        color="grey",
    )
    ax.errorbar(
        classical_pump_sep_thz[classical_blue_idxs],
        np.squeeze(ce_classical_mean[classical_blue_idxs, 0]),
        yerr=np.squeeze(ce_classical_std[classical_blue_idxs, 0]),
        capsize=capsize,
        label=r"Classical",
        fmt="-o",
        color="k",
    )
    ax.errorbar(
        qd_thz_sep[qd_blue_idxs],
        qd_blue_ces,
        yerr=qd_blue_stds,
        fmt="o",
        markersize=8,
        color=colors[0],
        capsize=8,
        label="Quantum",
    )
    ax.errorbar(
        qd_thz_sep[qd_red_idxs],
        qd_red_ces,
        yerr=qd_red_stds,
        fmt="o",
        markersize=8,
        capsize=8,
        color=colors[0],
        # label="Red-shifted idler",
    )
    if extra_data_idler_wls is not None:
        extra_data_ce_mean = cast(np.ndarray, extra_data_ce_mean)
        ax.plot(
            extra_data_idler_wls,
            extra_data_ce_mean,
            "-o",
            color="k",
            label=r"Unbalanced $P_\mathrm{q},P_\mathrm{p}$",
        )
    wl_lims = (np.min(idler_wls) - 1, np.max(idler_wls) + 1)
    thz_lims = (
        wls_to_thz(wl_lims[1], classical_pump_sep_thz, idler_wls),
        wls_to_thz(wl_lims[0], classical_pump_sep_thz, idler_wls),
    )
    ax.set_xlim(thz_lims)
    ax.set_ylim(0, 110)
    ax.set_xlabel(r"$\nu_\mathrm{i}-\nu_\mathrm{s}$ [THz]")
    ax.set_ylabel(r"$\eta_{\mathrm{peak}}$ [\%]")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    if not show_classical_error_bars:
        # remove ax errorbar classical entry from legend and add it back with different style
        handles, labels = ax.get_legend_handles_labels()
        classical_idx = labels.index("Classical")
        handles.pop(classical_idx)
        labels.pop(classical_idx)
        handles.insert(0, plt.plot([], [], "-o", color="k")[0])
        labels.insert(0, "Classical")

        ax.legend(handles, labels)
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            partial(thz_to_wls, thz_sep=classical_pump_sep_thz, idler_wls=idler_wls),
            partial(wls_to_thz, thz_sep=classical_pump_sep_thz, idler_wls=idler_wls),
        ),
    )
    secax.set_xlabel(r"$\lambda_\mathrm{i}$ [nm]", labelpad=10)
    secax.set_xticks(np.arange(965, 981, 5))
    # secax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    secax.set_xlim(
        thz_to_wls(thz_lims[0], classical_pump_sep_thz, idler_wls),
        thz_to_wls(thz_lims[-1], classical_pump_sep_thz, idler_wls),
    )

    ax.text(
        -0.18,
        1.08,
        r"\textbf{b}",
        transform=ax.transAxes,
        fontsize=48,
        va="top",
        ha="left",
    )
    return fig, ax


def plot_ce_vs_detuning_wl_bottom_ax(
    qd_thz_sep: np.ndarray,
    qd_ces: np.ndarray,
    qd_stds: np.ndarray,
    idler_wls_qd: np.ndarray,
    classical_pump_sep_thz: np.ndarray,
    ce_classical_mean: np.ndarray,
    ce_classical_std: np.ndarray,
    idler_wls: np.ndarray,
    colors: list,
    qd_red_idxs_to_ignore: np.ndarray = np.array([]),
    qd_blue_idxs_to_ignore: np.ndarray = np.array([]),
    show_classical_error_bars: bool = True,
    qd_as_rainbow: bool = False,
    extra_data_idler_wls: np.ndarray | None = None,
    extra_data_ce_mean: np.ndarray | None = None,
):
    classical_red_idxs, classical_blue_idxs = return_red_and_blue_idxs(
        classical_pump_sep_thz
    )
    qd_red_idxs, qd_blue_idxs = return_red_and_blue_idxs(qd_thz_sep)
    qd_red_idxs = np.setdiff1d(qd_red_idxs, qd_red_idxs_to_ignore + qd_red_idxs[0])
    qd_blue_idxs = np.setdiff1d(qd_blue_idxs, qd_blue_idxs_to_ignore + qd_blue_idxs[0])
    qd_blue_ces = qd_ces[qd_blue_idxs]
    qd_blue_stds = qd_stds[qd_blue_idxs]
    qd_red_ces = qd_ces[qd_red_idxs]
    qd_red_stds = qd_stds[qd_red_idxs]
    classical_red_idxs, classical_blue_idxs = return_red_and_blue_idxs(
        classical_pump_sep_thz
    )
    if show_classical_error_bars:
        capsize = 8
    else:
        capsize = 0
        ce_classical_std = np.zeros_like(ce_classical_std)
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    fig = cast(Figure, fig)

    ax.errorbar(
        idler_wls[classical_blue_idxs],
        np.squeeze(ce_classical_mean[classical_blue_idxs, 0]),
        yerr=np.squeeze(ce_classical_std[classical_blue_idxs, 0]),
        capsize=capsize,
        label=r"Classical",
        fmt="-o",
        color="k",
    )
    if not qd_as_rainbow:
        ax.errorbar(
            idler_wls_qd[qd_blue_idxs],
            qd_blue_ces,
            yerr=qd_blue_stds,
            fmt="o",
            markersize=8,
            color=colors[0],
            capsize=8,
            label="Quantum",
        )
        ax.errorbar(
            idler_wls_qd[qd_red_idxs],
            qd_red_ces,
            yerr=qd_red_stds,
            fmt="o",
            markersize=8,
            capsize=8,
            color=colors[0],
            # label="Red-shifted idler",
        )
    else:
        qd_thz_sep = qd_thz_sep[np.concatenate((qd_blue_idxs, qd_red_idxs))]
        idler_wls_qd = idler_wls_qd[np.concatenate((qd_blue_idxs, qd_red_idxs))]
        qd_ces = qd_ces[np.concatenate((qd_blue_idxs, qd_red_idxs))]
        qd_stds = qd_stds[np.concatenate((qd_blue_idxs, qd_red_idxs))]
        for qd_idx in range(len(qd_thz_sep)):
            ax.errorbar(
                idler_wls_qd[qd_idx],
                qd_ces[qd_idx],
                yerr=qd_stds[qd_idx],
                fmt="o",
                markersize=8,
                capsize=8,
                color=colors[qd_idx],
            )
        ax.errorbar(
            np.nan,
            np.nan,
            yerr=qd_stds[0],
            markersize=8,
            capsize=8,
            color="k",
            fmt="o",
            label="Quantum",
        )
    ax.errorbar(
        idler_wls[classical_red_idxs],
        np.squeeze(ce_classical_mean[classical_red_idxs, 0]),
        yerr=np.squeeze(ce_classical_std[classical_red_idxs, 0]),
        capsize=capsize,
        fmt="--o",
        color="k",
        label=r"Balanced $P_\mathrm{q},P_\mathrm{p}$",
    )
    ax.axvline(
        thz_to_wls(0, classical_pump_sep_thz, idler_wls),
        color="black",
        linestyle="--",
        alpha=0.7,
    )
    if extra_data_idler_wls is not None:
        extra_data_ce_mean = cast(np.ndarray, extra_data_ce_mean)
        ax.plot(
            extra_data_idler_wls,
            extra_data_ce_mean,
            "-o",
            color="k",
            # label=r"Unbalanced $P_\mathrm{q},P_\mathrm{p}$",
            zorder=10,
        )
    xlims = (np.min(idler_wls) - 1, np.max(idler_wls) + 1)
    ax.set_xlim(xlims)
    if np.max(ce_classical_mean) > 0:
        ax.set_ylim(0, 110)
    ax.set_xlabel(r"Output wavelength [nm]")
    ax.set_ylabel(r"$\eta_{\mathrm{peak}}$ [\%]")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    if not show_classical_error_bars:
        # remove ax errorbar classical entry from legend and add it back with different style
        handles, labels = ax.get_legend_handles_labels()
        classical_idx = labels.index("Classical")
        handles.pop(classical_idx)
        labels.pop(classical_idx)
        balanced_idx = labels.index("Balanced $P_\\mathrm{q},P_\\mathrm{p}$")
        handles.pop(balanced_idx)
        labels.pop(balanced_idx)
        handles.insert(0, plt.plot([], [], "-o", color="k")[0])
        labels.insert(0, "Classical")
        handles.append(plt.plot([], [], "--o", color="k")[0])
        labels.append(r"Balanced $P_\mathrm{q},P_\mathrm{p}$")
        ax.legend(handles, labels, loc="lower center")
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            partial(wls_to_thz, thz_sep=classical_pump_sep_thz, idler_wls=idler_wls),
            partial(thz_to_wls, thz_sep=classical_pump_sep_thz, idler_wls=idler_wls),
        ),
    )
    # secax.set_xlabel(r"$\nu_\mathrm{i}-\nu_\mathrm{s}$ [THz]", labelpad=10)
    secax.set_xlabel(r"$\Delta\nu$ [THz]", labelpad=10)
    secax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    # ax.text(
    #     -0.18,
    #     1.08,
    #     r"\textbf{b}",
    #     transform=ax.transAxes,
    #     fontsize=48,
    #     va="top",
    #     ha="left",
    # )
    return fig, ax
