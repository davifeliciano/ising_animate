"""
This is a module with an example of usage of the class Ising.
In this experiment, a given amount of 32 by 32 Ising models are
heaten up 0.1 degrees every 60 generations (an adiabatic heating)
from 1.0 to 7.0 degrees. This is useful to plot the thermodynamic 
quantities as functions of the temperature and show the phase transition 
at 2.27 degrees, in energy units. At the end, the average of the mean energy
and of the specific heat over the models are ploted in terms of the 
temperature, and the figure is shown and saved in the current directory.
"""

from dataclasses import dataclass, field
import multiprocessing as mp
import sys
import arrow

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from ..ising import Ising
from ..timer import timer


plt.rcParams.update(
    {
        "figure.autolayout": True,
        "figure.figsize": [9.6, 4.8],
        "axes.formatter.limits": (-3, 3),
    }
)


@dataclass
class PlotData:
    """Class to store data of the evolution of the
    macroscopic quantities during the heating process"""

    temp_data: "list[float]" = field(default_factory=list)
    mean_energy_data: "list[float]" = field(default_factory=list)
    specific_heat_data: "list[float]" = field(default_factory=list)


class HeatingIsing(Ising):
    """
    An example of usage of the Ising class. This is an Ising model
    that heat up 0.1 degrees every 60 generations (an adiabatic heating).
    This object is useful to plot the thermodynamic quantities as functions
    of the temperature and show the phase transition at 2.27 degrees,
    in energy units.
    """

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
        """
        Update the Ising Model and increase the temperature in 0.1 degrees
        if the current generation is a multiple of 60
        """
        super().update()
        # Increase temperature in 0.1 every 60 generations
        if not self.gen % 60:
            self.temp += 0.1
            self.plot.temp_data.append(self.temp)
            self.plot.mean_energy_data.append(np.mean(self.mean_energy_hist[-10:]))
            self.plot.specific_heat_data.append(np.mean(self.specific_heat_hist[-10:]))


def update_ising(ising):
    try:
        while ising.temp <= 7.0:
            ising.update()
        return ising
    # Ignore KeyboardInterrupt on a child process
    except KeyboardInterrupt:
        pass


@timer
def heatup_isings(
    ising_list: "list[HeatingIsing]", processes: int
) -> "list[HeatingIsing]":
    print(f"Heating up {processes} Ising Models. This may take some time...")
    with mp.Pool(processes) as pool:
        try:
            return pool.map(update_ising, ising_list)
        except KeyboardInterrupt:
            # Kill the pool when KeyboardInterrupt is raised
            pool.terminate()
            pool.join()
            sys.exit(1)


def pick_user_value():
    global cores
    while True:
        try:
            return int(input(f"Choose a value between 2 and {cores}: "))
        except ValueError:
            print("Invalid input! Try again...\n")
            return pick_user_value()


if __name__ == "__main__":
    # Ask user how many processes to use
    cores = mp.cpu_count()

    print(f"Your machine has {cores} logical processors.")
    print("How many processes do you want to create for the task?")
    print("Each process will consist of a heating Ising Model and")
    print("the results will be averaged over them.", end="\n")

    processes = pick_user_value()

    if processes < 2:
        print("\nToo few processes! Using 2 processes instead.")
        processes = 2
    elif processes > cores:
        print(f"\nToo many processes! Using {cores} processes instead.")
        processes = cores
    else:
        print("")

    # Create one instance of HeatingIsing for each process
    ising_list = [HeatingIsing(shape=(32, 32), temp=1.0) for _ in range(processes)]

    # Create processes an start computations
    print(f"Initializing process pool with {processes} processes.")
    mp.set_start_method("spawn")
    ising_list = heatup_isings(ising_list, processes)

    # Average data over the HeatingIsing instances
    temp_data = ising_list[0].plot.temp_data
    mean_energy_data = np.mean(
        [ising.plot.mean_energy_data for ising in ising_list], axis=0
    )
    specific_heat_data = (
        np.mean([ising.plot.specific_heat_data for ising in ising_list], axis=0)
        / ising_list[0].lattice.spins
    )

    # Create figure and axes to plot data
    fig, axes = plt.subplots(1, 2)

    mean_energy_filtered = gaussian_filter1d(mean_energy_data, sigma=2.0)
    specific_heat_filtered = gaussian_filter1d(specific_heat_data, sigma=2.0)

    fig.suptitle(
        "Specific Heat per spin and Magnetization in terms of Temperature\n"
        + f"Average of {processes} Ising Models"
    )

    axes[0].set(xlabel=r"$T$", ylabel=r"$\langle E \rangle$")
    axes[0].plot(temp_data, mean_energy_data, "ro", ms=3.0)
    axes[0].plot(temp_data, mean_energy_filtered, lw=1.7, zorder=-1)

    axes[1].set(xlabel=r"$T$", ylabel=r"$C / N$")
    axes[1].plot(temp_data, specific_heat_data, "ro", ms=3.0)
    axes[1].plot(temp_data, specific_heat_filtered, lw=1.7, zorder=-1)

    for ax in axes:
        ax.grid(linestyle=":")

    time_string = arrow.now().format("YYYY-MM-DD_HH-mm")
    fig_outfile = f"heating_{time_string}.png"
    print(f"Figure saved as {fig_outfile}")

    fig.savefig(fig_outfile)
    plt.show()
