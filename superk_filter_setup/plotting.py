import numpy as np
import matplotlib.pyplot as plt
import glob
from typing import cast

plt.style.use("custom")

data_dir = "./data"
files = glob.glob(data_dir + "/*.txt")
data = []
pow_per = 50
filtered_files = [file for file in files if f"{pow_per}per" in file]

for file in filtered_files:
    d = np.loadtxt(file)
    data.append(d)
fig, ax = plt.subplots()
ax = cast(plt.Axes, ax)
for idx_data, d in enumerate(data):
    filename = filtered_files[idx_data]
    d[1][d[1] < -80] = np.nan
    if "FES" in filename and "FEL" in filename:
        label = "FESH1000+FELH1500_R"
    elif "FEL" in filename:
        label = "FELH1500_R"
    else:
        label = "REF"
    ax.plot(
        d[0],
        d[1],
        label=label,
    )
ax.legend()
plt.show()
