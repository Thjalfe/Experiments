# CURRENTLY 4MSI PLOTTING OF PUMPS FOR DIFFERENT WAVELENGTHS AT THE EDGE OF THE BAND OF THE AMPLIFIERS
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.style.use("custom")
data_loc = "../data/4MSI/pump_specs/pump_specs_edfa_wl_limit.pkl"
with open(data_loc, "rb") as f:
    data = pickle.load(f)
fig_folder = "../../figs/4MSI/pump_specs/"
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

duty_cycle = data["dc"]
pump_input_powers = [key for key in data.keys() if isinstance(key, tuple)]
num_pump_wls = len(data[pump_input_powers[0]])
pump_wavelengths = np.array([[wl, 1604] for wl in np.arange(1609, 1609 + num_pump_wls)])
for pump_input_power in pump_input_powers:
    fig, ax = plt.subplots()
    spectra = data[pump_input_power]
    for i, spectrum in enumerate(spectra):
        ax.plot(spectrum[0, :], spectrum[1, :], label=f"{pump_wavelengths[i]} nm")
    ax.legend()
    ax.set_title(f"Pump input power: {pump_input_power} dBm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Power (dB)")
    plt.savefig(
        os.path.join(
            fig_folder, f"pump_specs_L_band_edge_Pin={pump_input_power}dBm.pdf"
        ),
        bbox_inches="tight",
    )
plt.show()
# |%%--%%| <2HQ45ZuQtu|2e2z0w6s5j>
data_loc = "../data/4MSI/pump_specs/only_1615_on.csv"
data = np.loadtxt(data_loc, delimiter=" ")
fig, ax = plt.subplots()
ax.plot(data[0, :], data[1, :])
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Power (dB)")
ax.set_title("Only 1615 nm pump on")
plt.savefig(os.path.join(fig_folder, "only_1615_on.pdf"), bbox_inches="tight")
plt.show()
# |%%--%%| <2e2z0w6s5j|fI9WdHGBv6>
data_loc = "../data/4MSI/pump_specs/CW_throughAOM_high_P_EDFA_after_tap_1604_1618_KK_at_1618.pkl"
with open(data_loc, "rb") as f:
    data = pickle.load(f)
data = data[:, data[1, :] > -75]
fig, ax = plt.subplots()
ax.plot(data[0, :], data[1, :])
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Power (dB)")
ax.set_title("Through AOM, max EDFA power after tap, w KK at 1618 nm")
plt.savefig(
    os.path.join(
        fig_folder, "CW_throughAOM_high_P_EDFA_after_tap_1604_1618_KK_at_1618.pdf"
    ),
    bbox_inches="tight",
)
plt.show()
