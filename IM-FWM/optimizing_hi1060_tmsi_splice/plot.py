# %%
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import cast
import numpy as np
import os
import glob

plt.style.use("custom")

file_names = glob.glob("./osa_time_meas/tisa_wl=950nm_2msi-len=100m/*.csv")
file_names = glob.glob("./osa_time_meas/tisa_wl=950nm_full_setup/*.csv")
fig_dir = "./figs"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
ref_file_names = [file_name for file_name in file_names if "ref" in file_name]
meas_file_names = [file_name for file_name in file_names if "ref" not in file_name]
ref_data = [np.loadtxt(file_name, delimiter=",") for file_name in ref_file_names]
meas_data = [np.loadtxt(file_name, delimiter=",") for file_name in meas_file_names]
fig, ax = plt.subplots()
ax = cast(Axes, ax)
for data in ref_data:
    ax.plot(data[0], data[1] - np.max(data[1]), label="Ref")
for idx_meas, data in enumerate(meas_data):
    label = f"Meas {idx_meas + 1}"
    if "SLA" in meas_file_names[idx_meas]:
        label = "With SLA"
    ax.plot(data[0], data[1] - np.max(data[1]), label=label)
ax.legend()
ax.set_xlabel("Time (s)")
ax.set_ylabel("Power (dB)")
fig.savefig(
    os.path.join(fig_dir, "osa-time-meas_before-after-sla-on-input.pdf"),
    bbox_inches="tight",
)
