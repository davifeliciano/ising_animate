import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({"figure.autolayout": True, "figure.figsize": [9.6, 4.8]})

data = pd.read_csv("data/heating.csv")

temp_data = data["temp"]
specific_heat_data = data["specific_heat"]
magnet_data = data["magnet"]

specific_heat_filtered = gaussian_filter1d(specific_heat_data, sigma=2.0)
magnet_filtered = gaussian_filter1d(magnet_data, sigma=2.0)

fig, ax = plt.subplots(1, 2)

fig.suptitle("Specific Heat per spin and Magnetization in terms of Temperature")

ax[0].set(xlabel=r"$T$", ylabel=r"$C / N$")
ax[0].plot(temp_data, specific_heat_data, "ro", ms=3.0)
ax[0].plot(temp_data, specific_heat_filtered, lw=1.7, zorder=-1)

ax[1].set(xlabel=r"$T$", ylabel=r"$\langle \mu \rangle$")
ax[1].plot(temp_data, magnet_data, "ro", ms=3.0)
ax[1].plot(temp_data, magnet_filtered, lw=1.7, zorder=-1)

fig.savefig("images/heating.png")
plt.show()
