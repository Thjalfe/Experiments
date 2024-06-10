import numpy as np
import matplotlib.pyplot as plt
import glob
data_dir = "../../data/modelocked_1571_pump/modelocked_as_pump/p2_wl=1563"
files = glob.glob(data_dir + "/*.csv")
all_data = []
for file in files:
    data = np.loadtxt(file, delimiter=",")
    all_data.append(data)
num_data_to_plot = 15
fig, ax = plt.subplots()
for data in all_data[:num_data_to_plot]:
    data[1, data[1, :] < -75] = np.nan
    ax.plot(data[0, :], data[1, :])
plt.show()
