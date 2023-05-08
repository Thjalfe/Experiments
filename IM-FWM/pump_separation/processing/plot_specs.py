import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("../")
from funcs.utils import process_dataset

plt.style.use("custom")
plt.ion()
file1 = np.loadtxt("../data/forlars/20dutycycle_2us.csv", delimiter=",")
file2 = np.loadtxt("../data/forlars/50dutycycle_5us.csv", delimiter=",")
file3 = np.loadtxt("../data/forlars/50dutycycle_3us.csv", delimiter=",")
# set all values smaller than -80 to nan in [1, :]
file1[file1[:, 1] < -80, 1] = np.nan
file2[file2[:, 1] < -80, 1] = np.nan
file3[file3[:, 1] < -80, 1] = np.nan
prominence = 0.1
height = -75
tolerance_nm = 0.25
max_peak_min_height = -35
file1_process = process_dataset(
    np.array([file1]),
    prominence,
    height,
    (1570, 1572),
    tolerance_nm,
    max_peak_min_height,
)
file2_process = process_dataset(
    np.array([file2]),
    prominence,
    height,
    (1570, 1572),
    tolerance_nm,
    max_peak_min_height,
)
file3_process = process_dataset(
    np.array([file3]),
    prominence,
    height,
    (1570, 1572),
    tolerance_nm,
    max_peak_min_height,
)

# |%%--%%| <Vwh3kx61kY|aiyQYqpt85>
CE1 = np.min(file1_process[0]['differences'])
CE2 = np.min(file2_process[0]['differences'])
CE3 = np.min(file3_process[0]['differences'])
plt.figure()
plt.plot(file1[:, 0], file1[:, 1], label=fr"20\% duty cycle, 2$\mu$s, CE={CE1:.2f} dB")
plt.plot(file2[:, 0], file2[:, 1], label=fr"50\% duty cycle, 5$\mu$s, CE={CE2:.2f} dB")
plt.plot(file3[:, 0], file3[:, 1], label=fr"50\% duty cycle, 3$\mu$s, CE={CE3:.2f} dB")
plt.legend(loc='best')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power (dBm)")
plt.savefig("all_three.png", bbox_inches="tight", dpi=300)

# plt.figure()
# plt.plot(file1[:, 0], file1[:, 1], label=r"20\% duty cycle, 2$\mu$s")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Power (dBm)")
# # plt.savefig("20dutycycle_2us.png", bbox_inches="tight")

# plt.figure()
# plt.plot(file2[:, 0], file2[:, 1], label=r"50\% duty cycle, 5$\mu$s")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Power (dBm)")
# # plt.savefig("50dutycycle_5us.png", bbox_inches="tight")

# plt.figure()
# plt.plot(file3[:, 0], file3[:, 1], label=r"50\% duty cycle, 3$\mu$s")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Power (dBm)")
# plt.savefig("50dutycycle_3us.png", bbox_inches="tight")
