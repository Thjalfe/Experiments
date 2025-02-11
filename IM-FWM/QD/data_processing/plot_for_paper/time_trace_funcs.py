import numpy as np


def find_start_idxs(
    time_arr: np.ndarray,
    voltage_arr: np.ndarray,
    rise_consistency: int,
    dc: float,
    pulse_freq: float = 1e5,
    min_group_size_for_consistent_diff: int = 5,
) -> np.ndarray:
    min_dist_between_pulse_starts = dc / pulse_freq
    min_idx_dist_between_pulse_starts = int(
        min_dist_between_pulse_starts / (time_arr[1] - time_arr[0])
    )
    diff = np.diff(voltage_arr)
    increasing = diff > 0
    possible_start_indices = [0]
    idx_counter = 0
    for i in range(len(increasing) - rise_consistency):
        if all(increasing[i : i + rise_consistency]):
            if (
                i - possible_start_indices[idx_counter]
                > min_idx_dist_between_pulse_starts
            ):
                possible_start_indices.append(i)
                idx_counter += 1
    pos_start_idxs_diff = np.diff(possible_start_indices)
    valid_diffs = np.where(pos_start_idxs_diff > min_idx_dist_between_pulse_starts)[0]
    group_starts = []
    current_group = [valid_diffs[0]]

    for i in range(1, len(valid_diffs)):
        if valid_diffs[i] == valid_diffs[i - 1] + 1:
            current_group.append(valid_diffs[i])
        else:
            # If the group is long enough, save the first index of the group
            if len(current_group) >= min_group_size_for_consistent_diff:
                group_starts.append(possible_start_indices[current_group[0]])
            current_group = [valid_diffs[i]]
    # Handle the last group
    if len(current_group) >= min_group_size_for_consistent_diff:
        group_starts.append(possible_start_indices[current_group[0]])
    group_starts = np.array(group_starts)
    group_starts = group_starts[group_starts > 0]  # It should never start at 0
    possible_start_indices = possible_start_indices[1:]
    return np.array(possible_start_indices)


def find_end_idxs(
    start_idxs: np.ndarray,
    voltage_arr: np.ndarray,
    dc: float,
    thresh_factor: float = 0.35,
) -> np.ndarray:
    max_val_each_pulse = np.array(
        [
            np.max(
                voltage_arr[
                    start_idx : start_idx + int(len(voltage_arr) / len(start_idxs))
                ]
            )
            for start_idx in start_idxs
        ]
    )
    max_loc_each_pulse = np.array(
        [
            np.argmax(
                voltage_arr[
                    start_idx : start_idx + int(len(voltage_arr) / len(start_idxs))
                ]
            )
            + start_idx
            for start_idx in start_idxs
        ]
    )
    thresholds = thresh_factor * max_val_each_pulse * dc
    idxs_end = []
    for i, start_idx in enumerate(start_idxs):
        thresh = thresholds[i]
        # thresh = 0.016  # nice magic number found by inspection
        pos_end_idxs = np.where(
            voltage_arr[
                max_loc_each_pulse[i] : (
                    len(voltage_arr)
                    if i == len(max_loc_each_pulse) - 1
                    else max_loc_each_pulse[i] + len(voltage_arr) // len(start_idxs)
                )
            ]
            < thresh
        )[0]
        # If the end of the pulse is not found, the pulse is not included
        if len(pos_end_idxs) == 0:
            continue
        idx_end = pos_end_idxs[0] + max_loc_each_pulse[i]
        idxs_end.append(idx_end)
    idxs_end = np.array(idxs_end)
    return idxs_end


def extract_pulse_idxs_equal_len(
    start_idxs: np.ndarray, end_idxs: np.ndarray, voltage_arr: np.ndarray
) -> np.ndarray:
    if len(end_idxs) < len(start_idxs):
        start_idxs = np.delete(start_idxs, -1)
    # make sure that all pulses are of the same length, so just make all of them the length of the longest pulse
    pulse_lens = [
        end_idx - start_idx for start_idx, end_idx in zip(start_idxs, end_idxs)
    ]
    max_pulse_len = np.max(pulse_lens)
    new_end_idxs = start_idxs + max_pulse_len
    pulse_idxs_tmp = [
        np.arange(start_idx, end_idx, 1)
        for start_idx, end_idx in zip(start_idxs, new_end_idxs)
    ]
    pulse_idxs = []
    for pulse_num, pulse_idxs_loc in enumerate(pulse_idxs_tmp):
        if any(pulse_idxs_loc >= len(voltage_arr)):
            continue
        pulse_idxs.append(pulse_idxs_loc)

    return np.array(pulse_idxs)


def correct_start_idxs(start_idxs: np.ndarray, voltage: np.ndarray) -> np.ndarray:
    pulse_start_vals = voltage[start_idxs]
    pulse_start_mean = np.mean(pulse_start_vals)
    corrected_start_idxs = []
    for start_idx in start_idxs:
        if voltage[start_idx] > pulse_start_mean:
            start_idx_copy = start_idx
            while voltage[start_idx_copy] > pulse_start_mean:
                start_idx_copy -= 1
            corrected_start_idxs.append(start_idx_copy)

        if voltage[start_idx] < pulse_start_mean:
            start_idx_copy = start_idx
            while voltage[start_idx_copy] < pulse_start_mean:
                start_idx_copy += 1
            corrected_start_idxs.append(start_idx_copy)
    return np.array(corrected_start_idxs)


def get_full_pulse_len_up_to_next_start(
    voltage: np.ndarray,
    time_before_next_pulse_start: float,
    pulse_window: float,
    pulse_idxs: np.ndarray,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gets full pulse up to start of next pulse instead of just getting around the pulse itself.
    Uses pulse_idxs as a starting point instead of start_idxs as the code was made earlier, so
    it was easier to just use pulse_idxs.
    """
    time_before_next_pulse_start = time_before_next_pulse_start * 1e-6
    time_before_next_pulse_start_idxs = int(time_before_next_pulse_start / time_step)
    pulse_window_idxs = int(pulse_window / time_step)
    pulses_full_duration_idxs = []
    # Goes to next pulse start and subtracts the time_before_next_pulse_start_idxs to get the full pulse
    for i in range(len(pulse_idxs)):
        if i != len(pulse_idxs) - 1:
            pulse_idxs_loc = [
                pulse_idxs[i, 0],
                pulse_idxs[i + 1, 0] - time_before_next_pulse_start_idxs,
            ]

        else:
            # The last pulse is handled differently, as there is no next pulse to go to
            # The window can be longer than a full pulse, so we need to check for that
            if pulse_idxs[i, -1] - pulse_idxs[i, 0] > (
                pulse_window_idxs - time_before_next_pulse_start_idxs
            ):
                pulse_idxs_loc = [
                    pulse_idxs[i, 0],
                    pulse_idxs[i, 0]
                    + (pulse_window_idxs - time_before_next_pulse_start_idxs),
                ]
                # If the pulse is shorter than the window, do nothing, as a later check will pad it with nan
            else:
                pulse_idxs_loc = [
                    pulse_idxs[i, 0],
                    pulse_idxs[i, -1],
                ]
        pulses_full_duration_idxs.append(pulse_idxs_loc)
    pulses_full_duration_idxs = np.array(pulses_full_duration_idxs)
    # Correct pulse lengths to be the same for all, they each have slightly different starting points
    mean_pulse_len = int(
        np.mean([pulse[1] - pulse[0] for pulse in pulses_full_duration_idxs])
    )
    pulses_full_duration = []
    for i in range(len(pulses_full_duration_idxs)):
        if i != len(pulses_full_duration_idxs) - 1:
            pulses_full_duration_idxs[i][1] = (
                pulses_full_duration_idxs[i][0] + mean_pulse_len
            )
        pulse_idxs_loc = pulses_full_duration_idxs[i]
        voltage_loc = voltage[pulse_idxs_loc[0] : pulse_idxs_loc[1]]
        if np.diff(pulse_idxs_loc) < mean_pulse_len:
            voltage_loc = np.pad(
                voltage_loc,
                (0, mean_pulse_len - int(np.diff(pulse_idxs_loc))),
                mode="constant",
                constant_values=np.nan,
            )
        else:
            voltage_loc = voltage_loc[:mean_pulse_len]
        pulses_full_duration.append(voltage_loc)
    pulses_full_duration = np.array(pulses_full_duration)
    time_full_duration = np.linspace(
        0, time_step * len(pulses_full_duration[0]), len(pulses_full_duration[0])
    )
    return time_full_duration, pulses_full_duration


def get_offset_val_to_zero_just_before_pulse(
    time: np.ndarray,
    voltage: np.ndarray,
    start_idxs: np.ndarray,
    time_before_pulse_start_for_mean: float = 1,
) -> np.floating:
    # time_before_pulse_start_for_mean is in micro seconds
    time_before_pulse_start_for_mean = time_before_pulse_start_for_mean * 1e-6
    num_points_before_pulse_start = int(
        time_before_pulse_start_for_mean / (time[1] - time[0])
    )
    voltages_before_pulses = [
        voltage[start_idx - num_points_before_pulse_start : start_idx]
        for start_idx in start_idxs
    ]
    offset_val = np.mean(voltages_before_pulses)
    return offset_val


def overlap_pulses(
    time: np.ndarray,
    voltage: np.ndarray,
    dc: float,
    pulse_freq: float = 1e5,
    smoothing_window: int = 100,
    rise_consistency: int = 50,
    time_before_next_pulse_start: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    trace = np.array([time, voltage])
    voltage_smoothed = np.convolve(
        trace[1, :], np.ones(smoothing_window) / smoothing_window, mode="same"
    )
    start_idxs = find_start_idxs(time, voltage_smoothed, rise_consistency, dc)
    start_idxs = correct_start_idxs(start_idxs, voltage_smoothed)
    noise_offset = get_offset_val_to_zero_just_before_pulse(
        time, voltage_smoothed, start_idxs
    )
    voltage_smoothed -= noise_offset
    voltage -= noise_offset
    end_idxs = find_end_idxs(start_idxs, voltage_smoothed, dc)
    pulse_idxs = extract_pulse_idxs_equal_len(start_idxs, end_idxs, voltage_smoothed)
    time_step = time[1] - time[0]
    pulse_window = 1 / pulse_freq
    time_full_duration, pulses_full_duration = get_full_pulse_len_up_to_next_start(
        voltage_smoothed,
        time_before_next_pulse_start,
        pulse_window,
        pulse_idxs,
        time_step,
    )

    pulses = np.array([voltage[pulse] for pulse in pulse_idxs])
    single_pulse_time_ax = np.linspace(0, time_step * len(pulses[0]), len(pulses[0]))
    return (
        single_pulse_time_ax,
        pulses,
        voltage_smoothed,
        time_full_duration,
        pulses_full_duration,
    )


def overlap_pulses_multiple_duty_cycles(
    time: np.ndarray,
    voltage: np.ndarray,
    duty_cycles: np.ndarray,
    ignore_dc_vals: list[float],
) -> dict:
    pulse_dict = {}
    for trace_num, dc in enumerate(duty_cycles):
        if dc in ignore_dc_vals:
            continue
        print(f"Processing duty cycle {dc}")
        pulse_dict[dc] = {}
        (
            single_pulse_time_ax,
            pulses,
            voltage_smoothed,
            time_full_duration,
            pulses_full_duration,
        ) = overlap_pulses(
            time[trace_num],
            voltage[trace_num],
            dc,
        )
        pulse_dict[dc]["single_pulse_time_ax"] = single_pulse_time_ax
        pulse_dict[dc]["duration"] = single_pulse_time_ax[-1]
        pulse_dict[dc]["pulses"] = pulses
        pulse_dict[dc]["pulses_full_duration"] = pulses_full_duration
        pulse_dict[dc]["time_full_duration"] = time_full_duration
        pulse_dict[dc]["time"] = time[trace_num]
        pulse_dict[dc]["voltage_smoothed"] = voltage_smoothed
        pulse_dict[dc]["mean_pulse"] = np.mean(pulses, axis=0)
        pulse_dict[dc]["std_pulse"] = np.std(pulses, axis=0)
        pulse_dict[dc]["mean_pulse_full_duration"] = np.nanmean(
            pulses_full_duration, axis=0
        )
        pulse_dict[dc]["std_pulse_full_duration"] = np.nanstd(
            pulses_full_duration, axis=0
        )
    return pulse_dict
