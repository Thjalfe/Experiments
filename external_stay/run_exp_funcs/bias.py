import time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from clients.dc_power_supply_clients import E3631A
from clients.power_meter_clients import Agilent8163x
from matplotlib.axes import Axes
from processing.helper_funcs import dBm_to_mW


def get_voltage_sign(dc: E3631A):
    if dc.cur_channel == 3:
        return -1
    else:
        return 1


def sweep_bias(
    voltage_arr: np.ndarray, pm: Agilent8163x, dc: E3631A, flip_voltage: int
):
    mzm_array = np.zeros((2, len(voltage_arr)))
    for i, voltage in enumerate(voltage_arr):
        dc.voltage = voltage * flip_voltage
        time.sleep(0.1)
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
