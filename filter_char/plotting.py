import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("large_fonts")

thorlabs_fesh1000_data = pd.read_excel("./data/FESH1000_OD.xlsx", skiprows=0)
thorlabs_fesh1000_data = thorlabs_fesh1000_data.drop(
    columns=["Unnamed: 0", "Unnamed: 1"]
)
thorlabs_fesh1000_data["OD"] = -np.log10(thorlabs_fesh1000_data["% Transmission"] / 100)
thorlabs_dmsp1500_data = pd.read_excel("./data/DMSP1500.xlsx", skiprows=1)
thorlabs_dmsp1500_data = thorlabs_dmsp1500_data.drop(
    columns=["Unnamed: 0", "Unnamed: 1"]
)
thorlabs_dmsp1500_data["Reflectance (dB)"] = -10 * np.log10(
    thorlabs_dmsp1500_data["% Reflectance"] / 100
)
thorlabs_dmsp1500_data["Optical Density"] = -np.log10(
    thorlabs_dmsp1500_data["% Transmission"] / 100
)
fig, ax = plt.subplots()
ax.plot(
    thorlabs_dmsp1500_data["Wavelength (nm)"],
    thorlabs_dmsp1500_data["% Transmission"],
    label="Transmission (%)",
)
ax.plot(
    thorlabs_dmsp1500_data["Wavelength (nm)"],
    thorlabs_dmsp1500_data["% Reflectance"],
    label="Reflectance (%)",
)
ax.set_xlabel("Wavelength (nm)")
ax.legend()
fig, ax = plt.subplots()
ax.plot(
    thorlabs_dmsp1500_data["Wavelength (nm)"],
    thorlabs_dmsp1500_data["Optical Density"],
)
# ax.plot(thorlabs_dmsp1500_data["Wavelength (nm)"], thorlabs_dmsp1500_data["Reflectance (dB)"], label="Reflectance (dB)")
ax.set_xlim(700, 2200)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("OD")
ax.set_title("Optical Density of Thorlabs DMSP1500")

fig, ax = plt.subplots()
ax.plot(thorlabs_fesh1000_data["Wavelength (nm)"], thorlabs_fesh1000_data["OD"])
ax.set_xlim(700, 2000)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("OD")
ax.set_title("Optical Density of Thorlabs FESH1000")
plt.show()
# |%%--%%| <TpW5dyeQvS|LUapmCQQb3>
### Edmund optics data
edmund_1300_SP = pd.read_excel(
    "./data/edmund_1300SP.xlsx",
    skiprows=0,
    names=["Wavelength (nm)", "Transmission (%)", "OD"],
)
edmund_1200_SP = pd.read_excel(
    "./data/edmund_1200SP.xlsx",
    skiprows=0,
    names=["Wavelength (nm)", "Transmission (%)", "OD"],
)
edmund_1100_SP = pd.read_excel(
    "./data/edmund_1100SP.xlsx",
    skiprows=0,
    names=["Wavelength (nm)", "Transmission (%)"],
)
edmund_1100_SP["OD"] = -np.log10(edmund_1100_SP["Transmission (%)"] / 100)
fig, ax = plt.subplots()
ax.plot(
    edmund_1300_SP["Wavelength (nm)"], edmund_1300_SP["OD"], label="1300SP", zorder=11
)
# ax.plot(edmund_1200_SP["Wavelength (nm)"], edmund_1200_SP["OD"], label="1200SP")
# ax.plot(edmund_1100_SP["Wavelength (nm)"], edmund_1100_SP["OD"], label="1100SP")
ax.set_xlim(1200, 1900)
ax.set_ylim(0, 10)
# ax.fill_between(
#     [1535, 1612], 0, 10, color="gray", alpha=0.5, label="C+L band amps", zorder=10
# )
ax.fill_between([1570, 1611], 0, 10, color="k", alpha=0.5, zorder=10)
# ax.legend()
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Optical Density")
# ax.set_title("Optical Density of Edmund Optics SP filters")
fig.save_figs(
    "../../papers/FC_QD_supplementary/figs/edmund_optics_SP_filter.pdf",
    bbox_inches="tight",
)
plt.show()
# |%%--%%| <LUapmCQQb3|WZIY23TXas>
bandpass = pd.read_excel("./data/FBH950-10.xlsx", skiprows=0)
fig, ax = plt.subplots()
ax.plot(bandpass["Wavelength (nm)"], bandpass["Optical Density"])
plt.show()
