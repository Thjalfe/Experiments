import numpy as np
import matplotlib.pyplot as plt
from typing import cast

filename = "./otdr-trace.dat"
data = np.loadtxt(filename, skiprows=0, delimiter="\t")
fig, ax = plt.subplots()
ax = cast(plt.Axes, ax)
ax.plot(data[:, 0], data[:, 1])
plt.show()
