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
        self.mean_energy_hist = [self.lattice.mean_energy()]
        self.magnet_hist = [self.lattice.magnet()]
        self.specific_heat_hist = [self.lattice.specific_heat()]
        self.susceptibility_hist = [self.lattice.susceptibility()]

    def update(self):
        self.mean_energy_hist.append(self.lattice.mean_energy())
        self.magnet_hist.append(self.lattice.magnet())
        self.specific_heat_hist.append(self.lattice.specific_heat())
        self.susceptibility_hist.append(self.lattice.susceptibility())
        self.lattice.update()


class AnimatedIsing(Ising):
    def __init__(
        self,
        shape=(30, 30),
        temp=2,
        j=(1, 1),
        field=0,
        init_state="random",
    ) -> None:

        super().__init__(
            shape=shape,
            temp=temp,
            j=j,
            field=field,
            init_state=init_state,
        )
        self.fig, self.ax = plt.subplots()
        self.animation = FuncAnimation(
            self.fig, self.update_animation, interval=100, save_count=60
        )

    def update_animation(self, frame):
        self.ax.clear()
        self.ax.imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))
        self.update()


if __name__ == "__main__":
    ising = AnimatedIsing(shape=(128, 128))
    ising.animation.save("images/test.gif")
