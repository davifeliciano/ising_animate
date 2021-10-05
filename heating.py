from dataclasses import dataclass, field
import multiprocessing as mp
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from ising import Ising
import timer


plt.rcParams.update({"figure.autolayout": True, "figure.figsize": [9.6, 4.8]})


@dataclass
class PlotData:
    """Class to store data of the evolution of the
    macroscopic quantities during the heating process"""

    temp_data: list[float] = field(default_factory=list)
    specific_heat_data: list[float] = field(default_factory=list)
    magnet_data: list[float] = field(default_factory=list)


class HeatingIsing(Ising):
    def __init__(
        self,
        shape=(30, 30),
        temp=2,
        j=(1, 1),
        field=0,
        init_state="random",
    ) -> None:

        super().__init__(
            shape=shape, temp=temp, j=j, field=field, init_state=init_state
        )
        self.plot = PlotData()

    def update(self):
        super().update()
        # Increase temperature in 0.1 every 60 generations
        if not self.gen % 60:
            self.temp += 0.1
            self.plot.temp_data.append(self.temp)
            self.plot.specific_heat_data.append(np.mean(self.specific_heat_hist[-10:]))
            self.plot.magnet_data.append(np.mean(self.magnet_hist[-10:]))


def update_ising(ising):
    try:
        while ising.temp <= 4.0:
            ising.update()
        return ising
    # Ignore KeyboardInterrupt on a child process
    except KeyboardInterrupt:
        pass


@timer.timer
def heatup_isings(ising_list: list[HeatingIsing], processes: int) -> list[HeatingIsing]:
    print(f"Heating up {processes} Ising Models. This may take some time...")
    with mp.Pool(processes) as pool:
        try:
            return pool.map(update_ising, ising_list)
        except KeyboardInterrupt:
            # Kill the pool when KeyboardInterrupt is raised
            pool.terminate()
            pool.join()
            sys.exit(1)


if __name__ == "__main__":
    # Number of processes to use based on cpu_count of the machine
    processes = int(mp.cpu_count() / 2)
    ising_list = [HeatingIsing(shape=(32, 32), temp=1.0) for _ in range(processes)]

    print(f"Initializing process pool with {processes} processes.")
    mp.set_start_method("spawn")
    ising_list = heatup_isings(ising_list, processes)

    temp_data = ising_list[0].plot.temp_data
    specific_heat_data = (
        np.mean([ising.plot.specific_heat_data for ising in ising_list], axis=0)
        / ising_list[0].lattice.spins
    )
    magnet_data = np.mean([ising.plot.magnet_data for ising in ising_list], axis=0)

    # Creating a dataframe with heating data
    result = pd.DataFrame(
        [temp_data, specific_heat_data, magnet_data],
        index=("temp", "specific_heat", "magnet"),
    )

    outfile = "data/heating.csv"
    result.transpose().to_csv(
        outfile,
        index=None,
        float_format="%.3f",
    )

    print(f"Results saved as {outfile}")

    # Ploting heating data
    fig, axes = plt.subplots(1, 2)

    specific_heat_filtered = gaussian_filter1d(specific_heat_data, sigma=2.0)
    magnet_filtered = gaussian_filter1d(magnet_data, sigma=2.0)

    fig.suptitle(
        "Specific Heat per spin and Magnetization in terms of Temperature\n"
        + f"Average of {processes} Ising Models"
    )

    axes[0].set(xlabel=r"$T$", ylabel=r"$C / N$")
    axes[0].plot(temp_data, specific_heat_data, "ro", ms=3.0)
    axes[0].plot(temp_data, specific_heat_filtered, lw=1.7, zorder=-1)

    axes[1].set(xlabel=r"$T$", ylabel=r"$\langle \mu \rangle$")
    axes[1].plot(temp_data, magnet_data, "ro", ms=3.0)
    axes[1].plot(temp_data, magnet_filtered, lw=1.7, zorder=-1)

    for ax in axes:
        ax.grid(linestyle=":")

    fig_outfile = "images/heating.png"
    print(f"Figure saved as {fig_outfile}")
    fig.savefig(fig_outfile)
    plt.show()
