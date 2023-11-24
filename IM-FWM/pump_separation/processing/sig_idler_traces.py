import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("custom")
plt.rcParams["figure.figsize"] = (20, 11)
plt.ion()
sig_loc = "../data/scope_traces/sig_idler_traces/pump_sep_15_sig.pkl"
idler_loc = "../data/scope_traces/sig_idler_traces/pump_sep_15_idler.pkl"
with open(sig_loc, "rb") as f:
    sig = pickle.load(f)
with open(idler_loc, "rb") as f:
    idler = pickle.load(f)

fig, ax = plt.subplots()
ax.plot(sig[0, :] * 10**6, sig[1, :] * 1000, label="Signal")
ax.plot(idler[0, :] * 10**6, idler[1, :] * 1000, label="Idler")
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (mV)")
ax.set_xlim([0, 40])
ax.legend()
ax.set_title(r"10\% duty cycle")
fig.savefig("../figs/pulse_shapes/first_sig_idler_trace.pdf", bbox_inches="tight")
