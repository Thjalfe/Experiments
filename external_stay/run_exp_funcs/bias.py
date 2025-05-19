import os
import pickle
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
import matplotlib.pyplot as plt
import numpy as np
from clients.dc_power_supply_clients import E3631A
from clients.oscilloscope_clients import HP83480
from clients.power_meter_clients import Agilent8163x
from matplotlib.axes import Axes
from external_stay.processing.helper_funcs import dBm_to_mW
from external_stay.run_exp_funcs.helper_funcs import (
    center_scope_around_peak,
    get_scope_data,
)


def get_voltage_sign(dc: E3631A):
    if dc.cur_channel == 3:
        return -1
    else:
        return 1


def sweep_bias(
    voltage_arr: np.ndarray,
    pm: Agilent8163x,
    dc: E3631A,
    flip_voltage: int,
    sleeptime: float = 0.1,
):
    mzm_array = np.zeros((2, len(voltage_arr)))
    for i, voltage in enumerate(voltage_arr):
        dc.voltage = voltage * flip_voltage
        time.sleep(sleeptime)
        mzm_array[0, i] = dc.voltage
        mzm_array[1, i] = pm.power
    return mzm_array


def sweep_all_dc_supplies(
    mzm_dict: dict,
    dc_lst: list,
    pms: list[Agilent8163x],
    dc_channels: list[int],
    voltages: np.ndarray,
):
    for i in range(len(dc_channels)):
        dc = dc_lst[i]
        dc_channel = dc_channels[i]
        dc.cur_channel = dc_channel
        flip_voltage = get_voltage_sign(dc)
        pm = pms[i]
        mzm_array = sweep_bias(voltages, pm, dc, flip_voltage)
        mzm_dict[f"mzm_{i+1}"] = mzm_array
    return mzm_dict


def plot_mzm_biases(mzm_dict: dict, data_dir: str, save_data: bool, show_plots: bool):
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    ax.plot(mzm_dict["mzm_1"][0], dBm_to_mW(mzm_dict["mzm_1"][1]), label="MZM 1")
    ax.plot(mzm_dict["mzm_2"][0], dBm_to_mW(mzm_dict["mzm_2"][1]), label="MZM 2")
    ax.plot(
        np.abs(mzm_dict["mzm_3"][0]), dBm_to_mW(mzm_dict["mzm_3"][1]), label="MZM 3"
    )
    ax.set_xlabel(r"$\left|\mathrm{Voltage}\right|$ [V]")
    ax.set_ylabel("Power [mW]")
    ax.legend()
    if save_data:
        fig.savefig(f"{data_dir}/mzm_bias_sweep_lin.pdf", bbox_inches="tight")
    fig, ax = plt.subplots()
    ax.plot(mzm_dict["mzm_1"][0], mzm_dict["mzm_1"][1], label="MZM 1")
    ax.plot(mzm_dict["mzm_2"][0], mzm_dict["mzm_2"][1], label="MZM 2")
    ax.plot(np.abs(mzm_dict["mzm_3"][0]), mzm_dict["mzm_3"][1], label="MZM 3")
    ax.set_xlabel(r"$\left|\mathrm{Voltage}\right|$ [V]")
    ax.set_ylabel("Power [mW]")
    ax.legend()
    if save_data:
        fig.savefig(f"{data_dir}/mzm_bias_sweep_dB.pdf", bbox_inches="tight")
    if show_plots:
        plt.show()


@dataclass
class BiasSweepIdxDepVars:
    dc_channel: int
    scope_channel: int
    dc_start_voltage: float
    dc: E3631A


@dataclass
class BiasSweepGeneralVars:
    num_reps: int
    delta_voltage: float
    voltage_stepsize: float
    trace_name_lst: Sequence[str | None]

    def __post_init__(self):
        self.trace_data_dict = {
            "traces": {trace_name: None for trace_name in self.trace_name_lst},
            "voltage_arrays": [[], [], []],
            "num_reps": self.num_reps,
        }


def sweep_scope_multiple_reps(
    scope: HP83480, num_reps: int, scope_channel
) -> np.ndarray:
    trace_arr = []

    for _ in range(num_reps):
        trace = get_scope_data(scope, scope_channel)
        trace_arr.append(trace)
    return np.array(trace_arr)


def sweep_bias_get_scope_traces(
    scope: HP83480,
    configs: list[BiasSweepIdxDepVars],
    settings: BiasSweepGeneralVars,
    data_dir: str,
    filename: str,
    save_data: bool,
    sleeptime: float = 1,
    pulse_window=500e-9,
) -> dict:
    trace_data_dict = settings.trace_data_dict

    for idx, config in enumerate(configs):
        trace_key = settings.trace_name_lst[idx]
        if trace_key is None:
            continue
        center_voltage = config.dc_start_voltage
        delta_voltage = settings.delta_voltage
        num_reps = settings.num_reps
        voltage_stepsize = settings.voltage_stepsize
        dc = config.dc
        dc.cur_channel = config.dc_channel
        scope_channel = config.scope_channel

        voltage_array = np.arange(
            center_voltage - delta_voltage,
            center_voltage + delta_voltage + delta_voltage / 10,
            voltage_stepsize,
        )
        voltage_array = (
            get_voltage_sign(dc) * voltage_array
        )  # if channel 3, voltage needs to be negative
        trace_data_dict["voltage_arrays"][idx] = voltage_array
        scope.timebase.scale = (pulse_window / 10) * 1.1  # Ensure the pulse is visible
        center_scope_around_peak(scope, channel=scope_channel, window_width_factor=3)
        dummy_trace = get_scope_data(scope, scope_channel)
        trace_data = np.zeros((len(voltage_array), num_reps, 2, len(dummy_trace[1])))
        for i, voltage in enumerate(voltage_array):
            dc.voltage = voltage
            time.sleep(sleeptime)
            trace_arr = sweep_scope_multiple_reps(scope, num_reps, scope_channel)
            trace_data[i] = trace_arr
            print(f"Done with measurement {i + 1}/{len(voltage_array)} for {trace_key}")
        trace_data_dict["traces"][trace_key] = trace_data
    if save_data:
        if f"{filename}.pkl" in os.listdir(data_dir):
            filename = f"{filename}_1"
        with open(f"{data_dir}/{filename}.pkl", "wb") as f:
            pickle.dump(trace_data_dict, f)
    return trace_data_dict


def get_dc_voltages(dc: E3631A, channels: list = [1, 2, 3]):
    volts = []
    for ch in channels:
        dc.cur_channel = ch
        volt = dc.voltage
        assert isinstance(volt, float)
        volts.append(float(np.round(volt, 3)))
    return volts


def set_dc_voltages(dc: E3631A, voltages: list, channels: list = [1, 2, 3]):
    for i, ch in enumerate(channels):
        dc.cur_channel = ch
        dc.voltage = voltages[i]


def increment_bias(dc: E3631A, channel: int, amount: float):
    dc.cur_channel = channel
    volt = dc.voltage
    assert isinstance(volt, float)
    dc.voltage = volt + amount


def start_scrambler(
    dc: E3631A,
    voltage_max: float = 5,
    voltage_stepsize: float = 0.1,
    sleep_between_increase: float = 0.05,
):
    # Hard check to see if it is the correct controller passed
    if dc.init_params["GPIB_address"] != 1:
        raise ValueError(
            "Wrong DC power supply connected, it should have GPIB address 1"
        )
    dc.cur_channel = 1
    cur_volt = dc.voltage
    assert isinstance(cur_volt, float)
    while cur_volt <= voltage_max:
        if cur_volt + voltage_stepsize > voltage_max:
            dc.voltage = voltage_max
            break
        cur_volt += voltage_stepsize

        dc.voltage = cur_volt
        time.sleep(sleep_between_increase)


def stop_scrambler(
    dc: E3631A,
    voltage_stepsize: float = -0.1,
    sleep_between_increase: float = 0.05,
):
    # Hard check to see if it is the correct controller passed
    if dc.init_params["GPIB_address"] != 1:
        raise ValueError(
            "Wrong DC power supply connected, it should have GPIB address 1"
        )
    dc.cur_channel = 1
    cur_volt = dc.voltage
    assert isinstance(cur_volt, float)
    while cur_volt > 0:
        cur_volt += voltage_stepsize
        dc.voltage = cur_volt
        time.sleep(sleep_between_increase)
        if cur_volt + voltage_stepsize < 0:
            dc.voltage = 0
            cur_volt = 0


def toggle_scrambling(dc: E3631A):
    cur_volt = dc.voltage
    assert isinstance(cur_volt, float)
    if cur_volt < 0.1:
        start_scrambler(dc)
    else:
        stop_scrambler(dc)
