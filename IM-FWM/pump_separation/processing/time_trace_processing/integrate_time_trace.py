import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simpson

file = "./idler_pump_normalized.csv"
df = pd.read_csv(file)
idler_time = df["Time"]
idler_voltage = df["Voltage"]
pump_time = df["Time.1"]
pump_voltage = df["Voltage.1"]
start_idx_idler = 50150
end_idx_idler = 55200
start_idx_pump = 50200
end_idx_pump = 55250
idler_time_subset = (
    idler_time[start_idx_idler:end_idx_idler] - idler_time[start_idx_idler]
)
idler_voltage_subset = idler_voltage[start_idx_idler:end_idx_idler]
pump_time_subset = pump_time[start_idx_pump:end_idx_pump] - pump_time[start_idx_pump]
pump_voltage_subset = pump_voltage[start_idx_pump:end_idx_pump]
pump_int = (
    simpson(pump_voltage_subset**2, x=pump_time_subset, dx=pump_time_subset.iloc[1])
    / pump_time_subset.iloc[-1]
)
idler_int = (
    simpson(idler_voltage_subset, x=idler_time_subset, dx=idler_time_subset.iloc[1])
    / idler_time_subset.iloc[-1]
)
fig, ax = plt.subplots()
ax.plot(
    idler_time_subset,
    idler_voltage_subset,
    label="idler",
)
ax.plot(
    pump_time_subset,
    pump_voltage_subset**2,
    label="pump",
)
ax.set_title(f"pump_int: {pump_int:.2f}, idler_int: {idler_int:.2f}")
ax.legend()
plt.show()
# |%%--%%| <1FmNc2ojag|bIF3vk1eIZ>
