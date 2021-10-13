from math import exp, sin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.ticker import StrMethodFormatter

from .lattice import Lattice

plt.rcParams.update({"figure.autolayout": True})


class Ising:
    """
    The core implementation of the Ising Model. No animation here.

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

    gen : int
        the current generation of the system. Starts at 0 and is
        incremented by a call to update method

    init_state : {"random", "down", "up"}; Default is "random"
        the initial configuration of the spins in the lattice.

    spins : int
        the number of the spins in the lattice

    lattice : Lattice
        an instance of a Lattice object that describes the
        current state of the system

    temp : float
        the current temperature of the lattice, in energy units.
        A new value can be assigned anytime.

    field : float
        the current value of the external magnetic field, oriented
        perpendicularlly to the lattice. A positive value represents
        a up oriented field. A new value can be assigned anytime.

    energy : float
        the total energy of the lattice in its current generation.

    mag_mom : float
        the total magnetic moment of the lattice in its current generation.

    mean_energy_hist : list[float]
        a list with the values of the mean energy for each past generation.
        New values are appended by a call to the update function

    magnet_hist : list[float]
        a list with the values of the magnetization per spin
        for each past generation. New values are appended by a call to
        the update function

    specific_heat_hist : list[float]
        a list with the values of the specific heat per spin
        for each past generation. New values are appended by a call to
        the update function

    susceptibility_hist : list[float]
        a list with the values of the magnetic susceptibility
        for each past generation. New values are appended by a call to
        the update function

    """

    def __init__(
        self,
        shape=(128, 128),
        temp=2.0,
        j=(1.0, 1.0),
        field=0.0,
        init_state="random",
    ) -> None:

        # Saving given init_state to include in __str__
        if init_state == "up":
            self._init_state = "up"
        elif init_state == "down":
            self._init_state = "down"
        else:
            self._init_state = "random"

        self.lattice = Lattice(shape, temp, j, field, init_state)
        self._energy = self.lattice.energy
        self._mag_mom = self.lattice.mag_mom
        self.mean_energy_hist = [self.lattice.mean_energy()]
        self.magnet_hist = [self.lattice.magnet() / self.spins]
        self.specific_heat_hist = [0.0]
        self.susceptibility_hist = [0.0]

    def __repr__(self) -> str:
        return (
            f"Ising(shape={self.lattice.shape.__str__()}, "
            + f"temp={self.lattice.temp}, "
            + f"j={self.lattice.j.__str__()}, "
            + f"field={self.lattice.field})"
        )

    def __str__(self) -> str:
        return (
            f"Ising Model with Temperature {self.lattice.temp:.2f} and Field {self.lattice.field:.2f}, "
            + f"starting with {self.init_state} spins"
        )

    @property
    def spins(self):
        return self.lattice.spins

    @property
    def init_state(self):
        return self._init_state

    @property
    def gen(self):
        return self.lattice._gen

    @property
    def temp(self):
        return self.lattice.temp

    @temp.setter
    def temp(self, value):
        self.lattice.temp = value

    @property
    def field(self):
        return self.lattice.field

    @field.setter
    def field(self, value):
        self.lattice.field = value

    @property
    def energy(self):
        return self.lattice.energy

    @property
    def mag_mom(self):
        return self.lattice.mag_mom

    def update(self):
        """
        Updates the system to the next generation, appending new values to
        the history list of each phisical quantity.
        """
        self.lattice.update()
        self.mean_energy_hist.append(self.lattice.mean_energy())
        self.magnet_hist.append(self.lattice.magnet() / self.spins)
        self.specific_heat_hist.append(self.lattice.specific_heat() / self.spins)
        self.susceptibility_hist.append(self.lattice.susceptibility())


class AnimatedIsing(Ising):
    """
    Animation of the Ising Model at constant temperature and
    constant external magnetic field

    Args
    ------------------

    shape : 2-tuple of ints. Default is (128, 128)
        the shape of the lattice of spins.

    temp : float. Default is 2.0
        the initial temperature of the lattice as a whole.

    j : float or 2-tuple of floats. Default is 1.0
        the coefficient of interaction between neighboring spins in the lattice.
        when a tuple is suplied, the first value is the coefficient for row neighbors
        and the second value is the coefficient for column neighbors.

    field : float. Default is 0.0
        the initial value for the external magnetic field.

    init_state : {"random", "down", "up"}. Default is "random"
        the initial configuration of the spins in the lattice.

    time_series : bool. Default is False
        wheater or not to include the time series of the macroscopic
        physical quantities in the animation

    interval : int. Default is 100
        the interval between each frame in the animation, in milliseconds

    frames : int. Default is 60
        the number of frames to include in the animation


    Attributes
    ------------------

    gen : int
        the current generation of the system. Starts at 0 and is
        incremented by a call to update method

    interval : int
        the interval between each frame in the animation, in milliseconds

    frames : int.
        the number of frames to include in the animation

    time : float
        elapsed time, considering one update per frame

    init_state : {"random", "down", "up"}; Default is "random"
        the initial configuration of the spins in the lattice.

    spins : int
        the number of the spins in the lattice

    lattice : Lattice
        an instance of a Lattice object that describes the
        current state of the system

    animation : FuncAnimation
        a matplotlib.animation.FuncAnimation object. The animation is saved
        with a call to animation.save("outfile.gif"). More info at
        https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html

    fig : Figure
        a matplotlib.figure.Figure object in which the animation takes place. More info at
        https://matplotlib.org/stable/api/figure_api.html?highlight=figure#matplotlib.figure.Figure

    ax : Axes or list of Axes
        single instance or list of matplotlib.axes.Axes objects. This are the axes in the
        figures in the animation. More info at
        https://matplotlib.org/stable/api/axes_api.html?highlight=axes#module-matplotlib.axes

    temp : float
        the current temperature of the lattice, in energy units.
        A new value can be assigned anytime.

    field : float
        the current value of the external magnetic field, oriented
        perpendicularlly to the lattice. A positive value represents
        a up oriented field. A new value can be assigned anytime.

    energy : float
        the total energy of the lattice in its current generation.

    mag_mom : float
        the total magnetic moment of the lattice in its current generation.

    time_hist : list[float]
        a list with the values of time for each past generation.
        New values are appended by a call to the update function

    mean_energy_hist : list[float]
        a list with the values of the mean energy for each past generation.
        New values are appended by a call to the update function

    magnet_hist : list[float]
        a list with the values of the magnetization per spin
        for each past generation. New values are appended by a call to
        the update function

    specific_heat_hist : list[float]
        a list with the values of the specific heat per spin
        for each past generation. New values are appended by a call to
        the update function

    susceptibility_hist : list[float]
        a list with the values of the magnetic susceptibility
        for each past generation. New values are appended by a call to
        the update function

    """

    def __init__(
        self,
        shape=(128, 128),
        temp=2,
        j=(1, 1),
        field=0,
        init_state="random",
        time_series=False,
        interval=100,
        frames=60,
    ) -> None:

        super().__init__(
            shape=shape,
            temp=temp,
            j=j,
            field=field,
            init_state=init_state,
        )

        self.time_series = bool(time_series)

        if self.time_series:
            self.fig, self.ax = plt.subplots(3, 2)
            self.fig.set_size_inches(10.8, 7.2)

            # Merging axis [0, 0] and [1, 0]
            gridspec = self.ax[0, 0].get_gridspec()
            for ax in self.ax[0:2, 0]:
                ax.remove()

            self.fig.add_subplot(gridspec[0:2, 0])
            self.ax = self.fig.get_axes()  # ax[0, 0] is now ax[4]
            self.ax.insert(0, self.ax.pop())  # ax[4] is now ax[0]

            self.__update_animation = self.__update_ani_time_series
            self.__init_animation = self.__init_ani_time_series
        else:
            self.fig, self.ax = plt.subplots()
            self.fig.set_size_inches(7.2, 4.8)

            self.__update_animation = self.__update_ani_no_time_series
            self.__init_animation = self.__init_ani_no_time_series

        self.fig.suptitle(self.__str__())

        self.interval = interval
        self.frames = frames

        self.time_hist = [self.time]

        self.animation = FuncAnimation(
            self.fig,
            func=self.__update_animation,
            init_func=self.__init_animation,
            interval=self.interval,
            save_count=self.frames,
        )

        self.axes_labels = {
            "time": r"$t$",
            "energy": r"$\langle E \rangle$",
            "magnet": r"$\langle \mu \rangle / n$",
            "specific_heat": r"$C / n$",
            "susceptibility": r"$\chi$",
        }

    def __repr__(self) -> str:
        return (
            f"AnimatedIsing(shape={self.lattice.shape.__str__()}, "
            + f"temp={self.lattice.temp}, "
            + f"j={self.lattice.j.__str__()}, "
            + f"field={self.lattice.field}, "
            + f"time_series={self.time_series}, "
            + f"interval={self.interval}, "
            + f"frames={self.frames})"
        )

    @property
    def time(self):
        return self.gen * self.interval / 1000

    def update(self):
        """
        Updates the system to the next generation, appending new values to
        the history list of each phisical quantity. This function is automatically
        called by the animation attribute to render the next frame.
        """
        super().update()
        self.time_hist.append(self.time)

    def __set_axes(self):
        for ax in self.ax[1:]:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
            ax.set(
                xlim=(0, self.frames * self.interval / 1000),
                xlabel=self.axes_labels["time"],
            )
            ax.grid(linestyle=":")

        self.ax[0].set(ylabel="i", xlabel="j")
        self.ax[1].set(ylabel=self.axes_labels["energy"])
        self.ax[2].set(ylabel=self.axes_labels["magnet"])
        self.ax[3].set(ylabel=self.axes_labels["specific_heat"])
        self.ax[4].set(ylabel=self.axes_labels["susceptibility"])

    def __init_ani_time_series(self):
        self.__set_axes()
        self.ax[0].imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))

    def __update_ani_time_series(self, frame):
        for ax in self.ax:
            ax.clear()

        self.update()
        self.__set_axes()
        self.fig.suptitle(self.__str__())
        self.ax[0].imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))
        self.ax[1].plot(self.time_hist, self.mean_energy_hist, color="purple")
        self.ax[2].plot(self.time_hist, self.magnet_hist, color="purple")
        self.ax[3].plot(self.time_hist, self.specific_heat_hist, color="purple")
        self.ax[4].plot(self.time_hist, self.susceptibility_hist, color="purple")

    def __init_ani_no_time_series(self):
        self.ax.set(ylabel="i", xlabel="j")
        self.ax.imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))

    def __update_ani_no_time_series(self, frame):
        self.ax.clear()
        self.update()
        self.ax.set(ylabel="i", xlabel="j")
        self.fig.suptitle(self.__str__())
        self.ax.imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))


class CoolingAnimatedIsing(AnimatedIsing):
    """
    Animation of the Ising Model at constant external magnetic field,
    but with its temperature growing (or decaing) exponentially, given
    initial and target values.

    Args
    ------------------

    shape : 2-tuple of ints
        the shape of the lattice of spins. Default is (128, 128).

    temp : float. Default is 5.0
        the initial temperature of the lattice as a whole.

    final_temp : float. Default is 1.0
        the final temperature of the system.

    cooling_rate : float. Default is 0.5
        the rate with which the temperature will find its way to the final_temp.
        (1 / cooling_rate) is the amount of time (in seconds) it takes to achieve about
        63% of the way there.

    j : float or 2-tuple of floats. Default is (1.0, 1.0)
        the coefficient of interaction between neighboring spins in the lattice.
        when a tuple is suplied, the first value is the coefficient for row neighbors
        and the second value is the coefficient for column neighbors.

    field : float. Default is 0.0
        the initial value for the external magnetic field.

    init_state : {"random", "down", "up"}. Default is "random"
        the initial configuration of the spins in the lattice.

    time_series : bool. Default is False
        wheater or not to include the time series of the macroscopic
        physical quantities in the animation

    interval : int. Default is 100
        the interval between each frame in the animation, in milliseconds

    frames : int. Default is 60
        the number of frames to include in the animation


    Attributes
    ------------------

    gen : int
        the current generation of the system. Starts at 0 and is
        incremented by a call to update method

    init_state : {"random", "down", "up"}; Default is "random"
        the initial configuration of the spins in the lattice.

    spins : int
        the number of the spins in the lattice

    lattice : Lattice
        an instance of a Lattice object that describes the
        current state of the system

    animation : FuncAnimation
        a matplotlib.animation.FuncAnimation object. The animation is saved
        with a call to animation.save("outfile.gif"). More info at
        https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html

    fig : Figure
        a matplotlib.figure.Figure object in which the animation takes place. More info at
        https://matplotlib.org/stable/api/figure_api.html?highlight=figure#matplotlib.figure.Figure

    ax : Axes or list of Axes
        single instance or list of matplotlib.axes.Axes objects. This are the axes in the
        figures in the animation. More info at
        https://matplotlib.org/stable/api/axes_api.html?highlight=axes#module-matplotlib.axes

    init_temp : float
        the initial temperature of the lattice as a whole.

    temp : float
        the current temperature of the lattice, in energy units.
        A new value can be assigned anytime.

    final_temp : float
        the final temperature of the system.

    cooling_rate : float
        the rate with which the temperature will find its way to the final_temp.
        (1 / cooling_rate) is the amount of time (in seconds) it takes to achieve about
        63% of the way there.

    field : float
        the current value of the external magnetic field, oriented
        perpendicularlly to the lattice. A positive value represents
        a up oriented field. A new value can be assigned anytime.

    energy : float
        the total energy of the lattice in its current generation.

    mag_mom : float
        the total magnetic moment of the lattice in its current generation.

    mean_energy_hist : list[float]
        a list with the values of the mean energy for each past generation.
        New values are appended by a call to the update function

    magnet_hist : list[float]
        a list with the values of the magnetization per spin
        for each past generation. New values are appended by a call to
        the update function

    specific_heat_hist : list[float]
        a list with the values of the specific heat per spin
        for each past generation. New values are appended by a call to
        the update function

    susceptibility_hist : list[float]
        a list with the values of the magnetic susceptibility
        for each past generation. New values are appended by a call to
        the update function

    """

    def __init__(
        self,
        shape=(128, 128),
        temp=5,
        final_temp=1,
        cooling_rate=0.5,
        j=(1, 1),
        field=0,
        init_state="random",
        time_series=False,
        interval=100,
        frames=100,
    ) -> None:

        super().__init__(
            shape=shape,
            temp=temp,
            j=j,
            field=field,
            init_state=init_state,
            time_series=time_series,
            interval=interval,
            frames=frames,
        )

        self._init_temp = abs(float(self.temp))
        self._final_temp = abs(float(final_temp))
        self._cooling_rate = abs(float(cooling_rate))

    def __repr__(self) -> str:
        return (
            f"CoolingAnimatedIsing(shape={self.lattice.shape.__str__()}, "
            + f"temp={self.lattice.temp}, "
            + f"final_temp={self.final_temp}, "
            + f"cooling_rate={self.cooling_rate}, "
            + f"j={self.lattice.j.__str__()}, "
            + f"field={self.lattice.field}, "
            + f"time_series={self.time_series}, "
            + f"interval={self.interval}, "
            + f"frames={self.frames})"
        )

    @property
    def init_temp(self):
        return self._init_temp

    @property
    def final_temp(self):
        return self._final_temp

    @property
    def cooling_rate(self):
        return self._cooling_rate

    def update(self):
        """
        Updates the system to the next generation, appending new values to
        the history list of each phisical quantity. This function is automatically
        called by the animation attribute to render the next frame.
        """
        super().update()
        self.temp = self.final_temp + (self.init_temp - self.final_temp) * exp(
            -self.cooling_rate * self.time
        )


class DynamicAnimatedIsing(Ising):
    """
    Animation of the Ising Model with both temperature and external magnetic
    field varying as functions of time

    Args
    ------------------

    shape : 2-tuple of ints; Default is (128, 128)
        the shape of the lattice of spins.

    temp : callable; Default is lambda t: 2.0
        a real valued one variable function that describes the temperature in
        the interval [0, interval * frames / 1000]

    j : float or 2-tuple of floats; Default is 1.0
        the coefficient of interaction between neighboring spins in the lattice.
        when a tuple is suplied, the first value is the coefficient for row neighbors
        and the second value is the coefficient for column neighbors.

    field : callable; Default is lambda t: math.sin(t)
        a real valued one variable function that describes the external
        magnetic field in the interval [0, interval * frames / 1000]

    init_state : {"random", "down", "up"}; Default is "random"
        the initial configuration of the spins in the lattice.

    time_series : bool. Default is False
        wheater or not to include the time series of the macroscopic
        physical quantities in the animation

    interval : int. Default is 100
        the interval between each frame in the animation, in milliseconds

    frames : int. Default is 60
        the number of frames to include in the animation


    Attributes
    ------------------

    gen : int
        the current generation of the system. Starts at 0 and is
        incremented by a call to update method

    init_state : {"random", "down", "up"}; Default is "random"
        the initial configuration of the spins in the lattice.

    spins : int
        the number of the spins in the lattice

    lattice : Lattice
        an instance of a Lattice object that describes the
        current state of the system

    animation : FuncAnimation
        a matplotlib.animation.FuncAnimation object. The animation is saved
        with a call to animation.save("outfile.gif"). More info at
        https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html

    fig : Figure
        a matplotlib.figure.Figure object in which the animation takes place. More info at
        https://matplotlib.org/stable/api/figure_api.html?highlight=figure#matplotlib.figure.Figure

    ax : Axes or list of Axes
        single instance or list of matplotlib.axes.Axes objects. This are the axes in the
        figures in the animation. More info at
        https://matplotlib.org/stable/api/axes_api.html?highlight=axes#module-matplotlib.axes

    temp : float
        the current temperature of the lattice, in energy units.
        A new value can be assigned anytime.

    field : float
        the current value of the external magnetic field, oriented
        perpendicularlly to the lattice. A positive value represents
        a up oriented field. A new value can be assigned anytime.

    temp_func : callable
        the function passed as temp to the constructor

    field_func : callable
        the function passed as field to the constructor

    energy : float
        the total energy of the lattice in its current generation.

    mag_mom : float
        the total magnetic moment of the lattice in its current generation.

    mean_energy_hist : list[float]
        a list with the values of the mean energy for each past generation.
        New values are appended by a call to the update function

    magnet_hist : list[float]
        a list with the values of the magnetization per spin
        for each past generation. New values are appended by a call to
        the update function

    specific_heat_hist : list[float]
        a list with the values of the specific heat per spin
        for each past generation. New values are appended by a call to
        the update function

    susceptibility_hist : list[float]
        a list with the values of the magnetic susceptibility
        for each past generation. New values are appended by a call to
        the update function

    """

    def __init__(
        self,
        shape=(128, 128),
        temp: callable = lambda t: 2.0,
        j=(1, 1),
        field: callable = lambda t: sin(t),
        init_state="random",
        time_series=False,
        interval=100,
        frames=60,
    ) -> None:

        super().__init__(
            shape=shape,
            temp=temp(0),
            j=j,
            field=field(0),
            init_state=init_state,
        )

        self.temp_func = temp
        self.field_func = field

        self.time_series = bool(time_series)

        if self.time_series:
            self.fig, self.ax = plt.subplots(4, 2)
            self.fig.set_size_inches(10.8, 9.6)

            # Merging axis [0, 0] and [1, 0]
            gridspec = self.ax[0, 0].get_gridspec()
            for ax in self.ax[0:2, 0]:
                ax.remove()

            self.fig.add_subplot(gridspec[0:2, 0])
            self.ax = self.fig.get_axes()  # ax[0, 0] is now ax[4]
            self.ax.insert(0, self.ax.pop())  # ax[4] is now ax[0]

            self.__update_animation = self.__update_ani_time_series
            self.__init_animation = self.__init_ani_time_series
        else:
            self.fig, self.ax = plt.subplots(2, 2)
            self.fig.set_size_inches(10.8, 4.8)

            # Merging axes [0, 0] and [1, 0]
            gridspec = self.ax[0, 0].get_gridspec()
            for ax in self.ax[:, 0]:
                ax.remove()

            self.fig.add_subplot(gridspec[:, 0])
            self.ax = self.fig.get_axes()  # ax[0, 0] is now ax[2]
            self.ax.insert(0, self.ax.pop())  # ax[2] is now ax[0]

            self.__update_animation = self.__update_ani_no_time_series
            self.__init_animation = self.__init_ani_no_time_series

        self.fig.suptitle(self.__str__())

        self.interval = interval
        self.frames = frames

        self.time_hist = [self.time]
        self.temp_hist = [self.temp]
        self.field_hist = [self.field]

        self.animation = FuncAnimation(
            self.fig,
            func=self.__update_animation,
            init_func=self.__init_animation,
            interval=self.interval,
            save_count=self.frames,
        )

        self.axes_labels = {
            "time": r"$t$",
            "energy": r"$\langle E \rangle$",
            "magnet": r"$\langle \mu \rangle / n$",
            "specific_heat": r"$C / n$",
            "susceptibility": r"$\chi$",
            "temp": r"$T$",
            "field": r"$H_z$",
        }

    def __repr__(self) -> str:
        return (
            f"DynamicAnimatedIsing(shape={self.lattice.shape.__str__()}, "
            + f"temp={self.temp_func.__name__}, "
            + f"j={self.lattice.j.__str__()}, "
            + f"field={self.field_func.__name__}, "
            + f"time_series={self.time_series}, "
            + f"interval={self.interval}, "
            + f"frames={self.frames})"
        )

    @property
    def time(self):
        return self.gen * self.interval / 1000

    def update(self):
        """
        Updates the system to the next generation, appending new values to
        the history list of each phisical quantity. This function is automatically
        called by the animation attribute to render the next frame.
        """
        self.temp = self.temp_func(self.time)
        self.field = self.field_func(self.time)
        super().update()
        self.time_hist.append(self.time)
        self.temp_hist.append(self.temp)
        self.field_hist.append(self.field)

    def __set_axes_time_series(self):
        for ax in self.ax[1:]:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
            ax.set(
                xlim=(0, self.frames * self.interval / 1000),
                xlabel=self.axes_labels["time"],
            )
            ax.grid(linestyle=":")

        self.ax[0].set(ylabel="i", xlabel="j")
        self.ax[1].set(ylabel=self.axes_labels["energy"])
        self.ax[2].set(ylabel=self.axes_labels["magnet"])
        self.ax[3].set(ylabel=self.axes_labels["specific_heat"])
        self.ax[4].set(ylabel=self.axes_labels["susceptibility"])
        self.ax[5].set(ylabel=self.axes_labels["temp"])
        self.ax[6].set(ylabel=self.axes_labels["field"])

    def __init_ani_time_series(self):
        self.__set_axes_time_series()
        self.ax[0].imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))

    def __update_ani_time_series(self, frame):
        for ax in self.ax:
            ax.clear()

        self.update()
        self.__set_axes_time_series()
        self.fig.suptitle(self.__str__())
        self.ax[0].imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))
        self.ax[1].plot(self.time_hist, self.mean_energy_hist, color="purple")
        self.ax[2].plot(self.time_hist, self.magnet_hist, color="purple")
        self.ax[3].plot(self.time_hist, self.specific_heat_hist, color="purple")
        self.ax[4].plot(self.time_hist, self.susceptibility_hist, color="purple")
        self.ax[5].plot(self.time_hist, self.temp_hist, color="purple")
        self.ax[6].plot(self.time_hist, self.field_hist, color="purple")

    def __set_axes_no_time_series(self):
        for ax in self.ax[1:]:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
            ax.set(
                xlim=(0, self.frames * self.interval / 1000),
                xlabel=self.axes_labels["time"],
            )
            ax.grid(linestyle=":")

        self.ax[0].set(ylabel="i", xlabel="j")
        self.ax[1].set(ylabel=self.axes_labels["temp"])
        self.ax[2].set(ylabel=self.axes_labels["field"])

    def __init_ani_no_time_series(self):
        self.__set_axes_no_time_series()
        self.ax[0].imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))

    def __update_ani_no_time_series(self, frame):
        for ax in self.ax:
            ax.clear()

        self.update()
        self.__set_axes_no_time_series()
        self.fig.suptitle(self.__str__())
        self.ax[0].imshow(self.lattice.state, norm=Normalize(vmin=-1.0, vmax=1.0))
        self.ax[1].plot(self.time_hist, self.temp_hist, color="purple")
        self.ax[2].plot(self.time_hist, self.field_hist, color="purple")
