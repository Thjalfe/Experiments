import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.style.use("custom")
filename = "../data/walk-off/HNDS1615AAA-6-3-2/1525.0nm-1565.0nm_scope-traces_259m.pkl"
with open(f"{filename}", "rb") as handle:
    data = pickle.load(handle)

save_figs = True
fig_dir = "/home/thjalfe/Documents/PhD/logbook/2025/march/figs/week4"
fig, ax = plt.subplots()
for i, trace in enumerate(data["traces"]):
    ax.plot(trace[0] * 10**9, trace[1] / np.max(trace[1]))
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Voltage [a.u.]")
ax.set_title(
    rf'$\lambda$={data["wavelengths"][0]:.1f}-{data["wavelengths"][-1]:.1f}nm, rising time diff $\sim$ 0.03 ns, 258 m HNLF'
)
if save_figs:
    fig.savefig(f"{fig_dir}/walk-off.pdf", bbox_inches="tight")


plt.show()
