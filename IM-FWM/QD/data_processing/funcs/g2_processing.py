import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import cast, Callable


def downsample_by_n(data, n):
    trunc_len = (len(data) // n) * n
    data = data[:trunc_len]
    reshaped_data = data.reshape(-1, n, data.shape[1])

    time_diff_downsampled = reshaped_data[:, 0, 0]
    counts_downsampled = np.mean(reshaped_data[:, :, 1], axis=1)

    return np.column_stack((time_diff_downsampled, counts_downsampled))


def normalize_data(
    data: np.ndarray, factor_for_edge_of_dataset: int
) -> tuple[np.ndarray, float]:
    data_len = len(data)
    normalize_len = data_len // factor_for_edge_of_dataset
    normalize_factor = (
        float(np.mean(data[:normalize_len, 1]) + np.mean(data[-normalize_len:, 1])) / 2
    )
    return (
        np.column_stack((data[:, 0], data[:, 1] / normalize_factor)),
        normalize_factor,
    )


def simple_weighting_around_dip(data: np.ndarray, center_range: int, weight: float):
    weights = np.ones_like(data[:, 1])
    weight_range = np.arange(
        np.argmin(np.abs(data[:, 1])) - center_range,
        np.argmin(np.abs(data[:, 1])) + center_range,
    )
    weights[weight_range] = weight
    return weights


def find_bunching_ratio_from_ref(
    ref_data: np.ndarray, time_window: int, additional_ignore_window: list = [-4, 6]
) -> float:
    ref_data_window = np.logical_and(
        ref_data[:, 0] > -time_window, ref_data[:, 0] < time_window
    )
    ref_data_window_additional = np.logical_or(
        ref_data[:, 0] < additional_ignore_window[0],
        ref_data[:, 0] > additional_ignore_window[1],
    )
    ref_data_window = np.logical_and(ref_data_window, ref_data_window_additional)
    ref_data_close = ref_data[ref_data_window]
    _, ref_norm_factor = normalize_data(ref_data_close, 1)
    _, ref_norm_factor_full = normalize_data(ref_data, 200)
    return ref_norm_factor_full / ref_norm_factor


def exp_decay_bunching(t: np.ndarray, *beta: float) -> np.ndarray:
    a = beta[0]
    b = beta[1]
    c = beta[2]
    tau_sp = beta[3]
    tau_bun = beta[4]
    return (
        1 - (a * np.exp(-np.abs(t - b) / tau_sp)) + c * np.exp(-np.abs(t - b) / tau_bun)
    )


def exp_decay(t: np.ndarray, *beta: float) -> np.ndarray:
    a = beta[0]
    b = beta[1]
    c = beta[2]
    tau_sp = beta[3]
    return a - ((a - b) * np.exp(-np.abs(t - c) / tau_sp))


def nlinfit(
    time: np.ndarray,
    hist_norm: np.ndarray,
    fit_fun: Callable,
    beta0: list[float],
    weights: np.ndarray,
    max_iter: int = 10000,
    bounds: tuple[list, list] = ([], []),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    opt_result, cov_matrix = curve_fit(
        fit_fun,
        time,
        hist_norm,
        p0=beta0,
        sigma=weights,
        maxfev=max_iter,
        bounds=bounds,
    )
    residuals = hist_norm - fit_fun(time, *opt_result)
    J = np.zeros((len(time), len(opt_result)))
    eps = np.sqrt(np.finfo(float).eps)
    for i in range(len(opt_result)):
        delta = np.zeros_like(opt_result)
        delta[i] = eps
        J[:, i] = (
            fit_fun(time, *(opt_result + delta)) - fit_fun(time, *(opt_result - delta))
        ) / (2 * eps)
    return opt_result, residuals, J, cov_matrix


def cw_g2(
    time: np.ndarray,
    hist_norm: np.ndarray,
    weights: np.ndarray | None = None,
    initial_guess: list[float] = [1, 1, 0.5, 1, 50],
    fit_fun: Callable = exp_decay_bunching,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if weights is None:
        weights = np.ones_like(hist_norm)
    bounds = ([], [])
    max_iter = 10000
    fit_result, residuals, jacobian, cov_matrix = nlinfit(
        time,
        hist_norm,
        fit_fun,
        initial_guess,
        weights,
        max_iter=max_iter,
        bounds=bounds,
    )
    std = np.sqrt(np.diag(cov_matrix))
    norm_val = np.max(fit_fun(np.linspace(-1000, 1000, 2000), *fit_result))
    return fit_result, residuals, jacobian, cov_matrix, std, norm_val


def plot_g2(
    time: np.ndarray,
    time_fit: np.ndarray,
    hist_norm: np.ndarray,
    fit_result: np.ndarray,
    residuals: np.ndarray,
    ci: np.ndarray,
    exp_fun: Callable,
    markerstyle: str = "o",
    markersize: int = 2,
    plot_confidence: bool = True,
    plotting_style="markers",
) -> tuple[Figure, Axes]:
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
        ax.bar(time, hist_norm, width=0.1, label="Raw data", color="k")
        ax.grid(False)
    ax.plot(
        time_fit,
        exp_fun(time_fit, *fit_result),
        linewidth=3,
        label="Data fit",
        color="C3",
        linestyle="--",
    )
    if plot_confidence:
        lower_bound = exp_fun(time_fit, *(fit_result - ci))
        upper_bound = exp_fun(time_fit, *(fit_result + ci))
        ax.fill_between(
            time_fit, lower_bound, upper_bound, color="gray", alpha=0.3, label="95% CI"
        )
    ax.set_xlabel("Time delay (ns)")
    ax.set_ylabel(r"$g^{(2)}(\tau)$")
    g2 = 1 - fit_result[0] + fit_result[2]
    g2_err = np.sqrt(ci[0] ** 2 + ci[2] ** 2)
    tau_sp_err = np.sqrt(ci[3] ** 2)
    tau_bun_err = np.sqrt(ci[4] ** 2)
    # ax.set_title(
    #     rf"RMSE={np.sqrt(np.mean(residuals**2)):.3f},  g(2)(0)={g2:.3f}, $\pm$ {g2_err:.3f}, $\tau_{{\mathrm{{sp}}}}$={fit_result[3]:.3f}$\pm$ {tau_sp_err:.2f}, $\tau_{{\mathrm{{bun}}}}$={fit_result[4]:.3f} $\pm$ {tau_bun_err:.2f}"
    # )
    ax.set_title(rf"g(2)(0)={g2:.3f}, $\pm$ {g2_err:.3f}")
    ax.legend()
    return fig, ax


def plot_both_fits(
    time_fit_ref: np.ndarray,
    fit_result_ref: np.ndarray,
    time_fit_idler: np.ndarray,
    fit_result_idler: np.ndarray,
    exp_fun: Callable,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    fig = cast(Figure, fig)
    ax = cast(Axes, ax)
    ax.plot(
        time_fit_ref,
        exp_fun(time_fit_ref, *fit_result_ref),
        linewidth=2,
        label="Reference fit",
    )
    ax.plot(
        time_fit_idler,
        exp_fun(time_fit_idler, *fit_result_idler),
        linewidth=2,
        label="Idler fit",
    )
    ax.set_xlabel("Time delay (ns)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend()
    return fig, ax
