import numpy as np
from numpy.random import default_rng

rng = default_rng()


class Lattice:
    def __init__(
        self,
        shape=(30, 30),
        temp=2.0,
        j=(1.0, 1.0),
        field=0.0,
        init_state="random",
    ) -> None:

        self._gen = 0
        rows, cols = shape
        self.rows, self.cols = self.shape = abs(int(rows)), abs(int(cols))
        self._step = self.rows * self.cols
        self._temp = abs(temp)
        self._field = float(field)

        try:
            self.j_row, self.j_col = self.j = j
        except TypeError:
            self.j_row = self.j_col = self.j = j

        if init_state == "up":
            self.state = np.full(shape=self.shape, fill_value=1)
        elif init_state == "down":
            self.state = np.full(shape=self.shape, fill_value=-1)
        else:
            self.state = rng.choice((1, -1), size=self.shape)
        self.init_state = self.state

        self.energy = self.lattice_energy()
        self.mag_mom = self.lattice_mag_mom()
        self.energy_hist = [self.energy]
        self.mag_mom_hist = [self.mag_mom]

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, value):
        self._temp = abs(value)

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        self._field = float(value)

    def element_energy(self, row: int, col: int) -> float:
        """Returns the energy of a element of the lattice"""
        return -self.state[row, col] * (
            self.field
            + self.j_row * self.state[(row + 1) % self.rows, col]
            + self.j_row * self.state[(row - 1 + self.rows) % self.rows, col]
            + self.j_col * self.state[row, (col + 1) % self.cols]
            + self.j_col * self.state[row, (col - 1 + self.cols) % self.cols]
        )

    def lattice_energy(self):
        """Returns the total energy of the lattice"""
        result = 0
        for i in range(self.rows):
            for j in range(self.cols):
                result += self.element_energy(i, j)
        return result

    def mean_energy(self):
        """Returns the mean energy of the lattice"""
        return np.mean(self.energy_hist)

    def lattice_mag_mom(self):
        """Returns the magnetic moment of the lattice"""
        return self.state.sum()

    def magnet(self):
        """Returns the magnetization of the lattice, i. e. the average of mag_mom_hist"""
        return np.mean(self.mag_mom_hist)

    def specific_heat(self):
        """Returns the specific heat of the lattice"""
        return np.var(self.energy_hist) / self.temp ** 2

    def susceptibility(self):
        """Returns the magnetic susceptibility of the lattice"""
        return np.var(self.mag_mom_hist) / self.temp

    def update(self):
        """Updates the system using metropolis algorithm"""
        self.energy_hist.clear()
        self.mag_mom_hist.clear()
        self._gen += 1

        for _ in range(self._step):
            # Choose a random spin0 in the lattice
            i = rng.integers(self.rows)
            j = rng.integers(self.cols)

            # Compute the change on the internal energy
            delta_energy = -2 * self.element_energy(i, j)

            if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / self.temp):
                self.state[i, j] *= -1
                self.energy += delta_energy
                self.mag_mom += 2 * self.state[i, j]
                self.energy_hist.append(self.energy)
                self.mag_mom_hist.append(self.mag_mom)


if __name__ == "__main__":
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        lattice = Lattice(shape=(15, 15))
        for i in range(5):
            lattice.update()
            print(f"Generation = {i + 1}", end="\n\n")
            print(
                f"Energy History = {lattice.energy_hist}",
                f"Magnetic Moment = {lattice.mag_mom_hist}",
                sep="\n",
                end="\n\n",
            )
            print(
                f"Mean Energy = {lattice.mean_energy()}",
                f"Magnetization = {lattice.magnet()}",
                f"Specific Heat = {lattice.specific_heat()}",
                f"Susceptibility = {lattice.susceptibility()}",
                sep="\n",
                end="\n\n",
            )

    print(f"Energy History = ")

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
