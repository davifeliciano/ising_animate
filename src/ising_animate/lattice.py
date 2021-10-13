import numpy as np
from numpy.random import default_rng

rng = default_rng()


class Lattice:
    """
    A Lattice of spins evolving acording to the
    metropolis algorithm

    Args
    ------------------

    shape : 2-tuple of ints; Default is (128, 128)
        the shape of the lattice of spins.

    temp : float; Default is 2.0
        the initial temperature of the lattice as a whole.

    j : float or 2-tuple of floats; Default is 1.0
        the coefficient of interaction between neighboring spins in the lattice.
        when a tuple is suplied, the first value is the coefficient for row neighbors
        and the second value is the coefficient for column neighbors.

    field : float; Default is 0.0
        the initial value for the external magnetic field.

    init_state : {"random", "down", "up"}; Default is "random"
        the initial configuration of the spins in the lattice.

    Attributes
    ------------------

    rows : int
        number of rows of spins in the lattice

    cols : int
        number of columns of spins in the lattice

    spins : int
        total number of spins in the lattice.
        The product between rows and cols.

    temp : float
        the current temperature of the lattice, in energy units.
        A new value can be assigned anytime.

    field : float
        the current value of the external magnetic field, oriented
        perpendicularlly to the lattice. A positive value represents
        a up oriented field. A new value can be assigned anytime.

    state : numpy.ndarray
        an matrix with values representing the spins of the lattice.
        -1 represents the down configuration, while +1 represents the up
        configuration.

    init_state : numpy.ndarray
        an matrix with values representing the initial configuration
        of the lattice. -1 represents the down configuration,
        while +1 represents the up configuration.

    energy : float
        the total energy of the lattice in its current generation.

    mag_mom : float
        the total magnetic moment of the lattice in its current generation.

    energy_hist : list[float]
        a list with the values of the total energies of the lattice during each
        monte carlo step. This list is filled by a call to the update method,
        and its mean value corresponds to the mean energy of the lattice.

    mag_mom_hist : list[float]
        a list with the values of the magnetic moment of the lattice during each
        monte carlo step. This list is filled by a call to the update method,
        and its mean value corresponds to the magnetization of the lattice.
    """

    def __init__(
        self,
        shape=(128, 128),
        temp=2.0,
        j=(1.0, 1.0),
        field=0.0,
        init_state="random",
    ) -> None:

        self._gen = 0
        rows, cols = shape
        self._rows, self._cols = self.shape = abs(int(rows)), abs(int(cols))

        temp = abs(float(temp))
        if temp:
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

    def __repr__(self) -> str:
        return f"Lattice(shape={self.shape.__str__()}, temp={self.temp}, j={self.j.__str__()}, field={self.field})"

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def spins(self):
        return self.rows * self.cols

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, value):
        temp = abs(value)
        if temp:
            self._temp = abs(float(value))

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        self._field = float(value)

    def element_energy(self, row: int, col: int) -> float:
        """
        Returns the energy of an element of the lattice,
        given its row and column.
        """
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
        """
        Returns the magnetization of the lattice,
        i. e. the average of mag_mom_hist
        """
        return np.mean(self.mag_mom_hist)

    def specific_heat(self):
        """
        Returns the specific heat of the lattice, i. e.
        the variance of energy_hist
        """
        return np.var(self.energy_hist) / self.temp ** 2

    def susceptibility(self):
        """
        Returns the magnetic susceptibility of the lattice, i. e.
        the variance of the mag_mom_hist divided by the temp
        """
        return np.var(self.mag_mom_hist) / self.temp

    def update(self):
        """Updates the system using metropolis algorithm"""
        self.energy_hist.clear()
        self.mag_mom_hist.clear()
        self._gen += 1

        for _ in range(self.spins):
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
