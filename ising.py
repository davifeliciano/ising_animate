import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

from lattice import Lattice


class Ising:
    def __init__(
        self,
        shape=(30, 30),
        temp=2.0,
        j=(1.0, 1.0),
        field=0.0,
        init_state="random",
    ) -> None:

        self.lattice = Lattice(shape, temp, j, field, init_state)
        self.fig, self.ax = plt.subplots()
        self.animation = FuncAnimation(
            self.fig, self.update_animation, interval=100, save_count=60
        )

        self.mean_energy_hist = []
        self.magnet_hist = []
        self.specific_heat_hist = []
        self.susceptibility_hist = []

    def update_animation(self, frame):
        self.ax.clear()
        self.ax.imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))
        self.update()

    def update(self):
        self.mean_energy_hist.append(self.lattice.mean_energy())
        self.magnet_hist.append(self.lattice.magnet())
        self.specific_heat_hist.append(self.lattice.specific_heat())
        self.susceptibility_hist.append(self.lattice.susceptibility())
        self.lattice.update()


if __name__ == "__main__":
    ising = Ising(shape=(128, 128), temp=1.0, init_state="random")
    ising.animation.save("ising.gif")
