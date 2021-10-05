from dataclasses import dataclass, field
import multiprocessing as mp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ising import Ising
import timer


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
    while ising.temp <= 4.0:
        ising.update()
    return ising


@timer.timer
def heatup_isings(isings):
    with mp.Pool() as pool:
        print(f"Heating up {isings_number} Ising Models! This may take some time...")
        isings = pool.map(update_ising, isings)
    return isings


if __name__ == "__main__":
    # List with ising models to average over
    isings_number = 8
    isings = [HeatingIsing(shape=(64, 64), temp=1.0) for _ in range(isings_number)]

    print("Spawning procecesses...")
    mp.set_start_method("spawn", force=True)
    isings = heatup_isings(isings)

    temp_data = isings[0].plot.temp_data
    specific_heat_data = (
        np.mean([ising.plot.specific_heat_data for ising in isings], axis=0)
        / isings[0].lattice.spins
    )
    magnet_data = np.mean([ising.plot.magnet_data for ising in isings], axis=0)

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

    print(f"Results saved at {outfile}")

    fig, ax = plt.subplots(1, 2)

    ax[0].set(xlabel=r"$T$", ylabel=r"$C / N$")
    ax[0].plot(temp_data, specific_heat_data, "ro", ms=3.0)
    ax[1].set(xlabel=r"$T$", ylabel=r"$\langle \mu \rangle$")
    ax[1].plot(temp_data, magnet_data, "ro", ms=3.0)

    plt.show()
