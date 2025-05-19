import re
from external_stay.run_exp_funcs.helper_funcs import find_square_region
from external_stay.processing.helper_funcs import rolling_average, dBm_to_mW, mW_to_dBm
from typing import Any
import pickle
from dataclasses import dataclass
import numpy as np
import glob


def get_power_from_filename(filename: str):
    pattern = r"([+-]?\d*\.?\d+)(?=dBm)"
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"No power value found in the filename: {filename}")


def load_pickle(filename: str) -> Any:
    with open(f"{filename}", "rb") as handle:
        data = pickle.load(handle)
    return data


def load_dataset_multiple_powers(data_dir: str) -> tuple[list[dict], np.ndarray]:
    filenames = glob.glob(f"{data_dir}/*.pkl")
    powers = np.array([get_power_from_filename(f) for f in filenames])
    sort_idx = np.argsort(powers)
    return [load_pickle(filenames[i]) for i in sort_idx], powers[sort_idx]


@dataclass
class ProcessedTraces:
    raw: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    mean_normalized: np.ndarray
    std_normalized: np.ndarray
    pulse_edges_idx: tuple[int, int]
    pulse_edges_time: tuple[float, float]


@dataclass
class TimeTraceData:
    trace_names: list[str]
    time_axis: np.ndarray
    processed_traces: dict[str, ProcessedTraces]
    idler_loc_idxs: tuple[int, int]
    idler_loc_time: tuple[float, float]


def optimal_idler_loc_for_fwm(
    trace: np.ndarray, time_ax: np.ndarray, pulse_duration: float
):
    time_stepsize = time_ax[1] - time_ax[0]
    pulse_duration_idxs = int(pulse_duration / time_stepsize)
    window_sum = np.sum(trace[:pulse_duration_idxs])
    max_sum = window_sum
    max_start = 0
    for i in range(1, len(trace) - pulse_duration_idxs + 1):
        window_sum += trace[i + pulse_duration_idxs - 1] - trace[i - 1]
        if window_sum > max_sum:
            max_sum = window_sum
            max_start = i

    return max_start, max_start + pulse_duration_idxs


def is_pulse_square(
    pulse_edges: tuple[int, int],
    expected_duration: float,
    time_axis: np.ndarray,
    threshold: float = 0.95,
) -> bool:
    actual_duration = np.abs(time_axis[pulse_edges[1]] - time_axis[pulse_edges[0]])
    return (actual_duration / expected_duration) >= threshold


def find_triangle_pulse_edges(
    trace: np.ndarray, pulse_duration: float, time_ax: np.ndarray
) -> tuple[int, int]:
    pulse_center = np.argmax(trace)
    time_stepsize = time_ax[1] - time_ax[0]
    pulse_duration_idxs = int(pulse_duration / time_stepsize)
    left_idx = int(pulse_center - (pulse_duration_idxs / 2))
    right_idx = int(pulse_center + (pulse_duration_idxs / 2))
    return left_idx, right_idx


def process_single_dataset_entry(
    data: dict,
    pulse_duration: float,
    time_axis: np.ndarray,
    square_threshold: float = 0.95,
) -> TimeTraceData:

    processed_traces = {}
    for trace_name in data:
        trace_data = data[trace_name]["time_traces"]
        raw_traces = trace_data[:, 1, :]
        mean_trace = np.mean(raw_traces, axis=0)
        std_trace = np.std(raw_traces, axis=0)
        mean_normalized = mean_trace / np.max(mean_trace)
        std_normalized = std_trace / np.max(mean_trace)

        # Pulse shape analysis - only need one rolled for shape
        sample_trace = rolling_average(raw_traces[0])
        edges = find_square_region(sample_trace, threshold_ratio=0.7)

        if not is_pulse_square(edges, pulse_duration, time_axis, square_threshold):
            edges = find_triangle_pulse_edges(sample_trace, pulse_duration, time_axis)

        processed_traces[trace_name] = ProcessedTraces(
            raw=raw_traces,
            mean=mean_trace,
            std=std_trace,
            mean_normalized=mean_normalized,
            std_normalized=std_normalized,
            pulse_edges_idx=edges,
            pulse_edges_time=(time_axis[edges[0]], time_axis[edges[1]]),
        )
    idler_loc_idxs = optimal_idler_loc_for_fwm(
        mean_trace, time_axis, pulse_duration  # pyright: ignore
    )
    idler_loc_time = (time_axis[idler_loc_idxs[0]], time_axis[idler_loc_idxs[1]])
    return TimeTraceData(
        time_axis=time_axis,
        trace_names=list(data.keys()),
        processed_traces=processed_traces,
        idler_loc_idxs=idler_loc_idxs,
        idler_loc_time=idler_loc_time,
    )


def calc_gain(
    wl_ax: np.ndarray,
    ref_spectrum: np.ndarray,
    pump_spectrum: np.ndarray,
    amplified_spectrum: np.ndarray,
    osa_res_tol_factor: int = 5,
) -> tuple[float, float]:
    ref_spectrum = ref_spectrum[1]
    pump_spectrum = pump_spectrum[1]
    amplified_spectrum = amplified_spectrum[1]
    osa_res = np.round(wl_ax[1] - wl_ax[0], 3)
    wl_tol = osa_res_tol_factor * osa_res
    ref_powers = ref_spectrum
    sig_wl = wl_ax[np.argmax(ref_powers)]
    sig_power = np.max(ref_powers)
    wl_idxs_within_tol = np.where(
        (wl_ax > sig_wl - wl_tol) & (wl_ax < sig_wl + wl_tol)
    )[0]
    pump_power = np.max(pump_spectrum[wl_idxs_within_tol])
    amplified_power = np.max(amplified_spectrum[wl_idxs_within_tol])
    amplified_power_plus_pump_bg = mW_to_dBm(
        dBm_to_mW(amplified_power) + dBm_to_mW(pump_power)
    )
    gain = amplified_power_plus_pump_bg - sig_power
    return gain, amplified_power


@dataclass
class SpectraProcessed:
    wl_ax: np.ndarray
    signal_ref: np.ndarray
    pump_ref: np.ndarray
    amplified_spectrum: np.ndarray

    def __post_init__(self):
        self.gain, self.amplified_power = calc_gain(
            self.wl_ax, self.signal_ref, self.pump_ref, self.amplified_spectrum
        )
