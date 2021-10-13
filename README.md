# ising_animate
A Python Package to easily generate animations of the [Ising Model](https://en.wikipedia.org/wiki/Ising_model) 
using the [Metropolis Algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm), the most
commonly used Markov Chain Monte Carlo method to calculate estimations for this system.

![ising_2021-10-12_15-14-51](https://user-images.githubusercontent.com/26972046/137008265-33f7b181-7047-4afe-b044-ac5f856df73c.gif)
## Installation

Be sure to hav Python 3.6 or newer installed on your machine. Then you can simply use pip to install the package and its dependencies.
```
pip install ising_animate
```
## Usage

### Command Line Tool
This package can be used as a command line tool to generate the desired animations. For instance, the animation above was created using the command

```
python -m ising_animate
```
You can specify the desired size, initial state, temperature or field using some command line options. For example, the command
```
python -m ising_animate --size 256 --temp 1.5 --field 1.0 --init-state "down" --time-series
```
yields
![ising_2021-10-12_15-26-03](https://user-images.githubusercontent.com/26972046/137010154-bc7d30c0-7ab3-44a9-b8a4-8e76f3e5b2c7.gif)

For a full description of all the available options, type in ```python -m ising_animate --help```.
### Import
When imported, there are four classes of objects that can be used to create custom animations: 
* [Ising](https://davifeliciano.github.io/ising_animate/ising.html#ising_animate.ising.Ising): just the core implementation of the Ising Model, no animation;
* [AnimatedIsing](https://davifeliciano.github.io/ising_animate/ising.html#ising_animate.ising.AnimatedIsing): an animation of the Ising Model with both temperature and external magnetic field held at fixed values;
* [CoolingAnimatedIsing](https://davifeliciano.github.io/ising_animate/ising.html#ising_animate.ising.CoolingAnimatedIsing): an animation of the Ising Model with the temperature droping (or raising) exponentially to a target value, at a given rate;
* [DynamicAnimatedIsing](https://davifeliciano.github.io/ising_animate/ising.html#ising_animate.ising.DynamicAnimatedIsing): an animation of the Ising Model with temperature and external magnetic field both described by given functions of time.

The usage of all of them are very similar. You just have to create an instance with the desired arguments...
```python
from ising_animate import DynamicAnimatedIsing
from math import sin

frames = 100

dynamic = DynamicAnimatedIsing(
    shape=(256, 256),                # the shape of the lattice
    temp=lambda t: 1.0 + 0.3 * t,    # temperature as a function of time
    field=lambda t: sin(t),          # external magnetic field as a function of time
    time_series=True,                # plot evolution of physical quantities over time
    interval=100,                    # interval of each frame
    frames=frames,                   # amount of frames in the animation
)
```
and the animation itself is now given by a matplotlib [FuncAnimation](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html)
object stored at ```dynamic.animation```. This object has a [save](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.Animation.html#matplotlib.animation.Animation.save) 
method of which the most important arguments is the output filename and the fps of the animation. The fps can be chosen as you wish,
but the natural value is 1000 (the amount of milliseconds in one second) divided by the interval of each frame. So, in that case, fps is 10.
```python
dynamic.animation.save("outfile.gif", fps=10)
```
![outfile](https://user-images.githubusercontent.com/26972046/137047228-f8a0f75c-fbae-4320-8416-c1aff0503548.gif)
Another useful feature of the save method is the possibility to pass a progress callback function that
will be called after drawing each frame, with the count of already drawn frames and the total number of frames on
the animation. This makes easy to use a package like progressbar2 to show the progress in a progress bar.
```python
import progressbar

with progressbar.ProgressBar(max_value=frames) as bar:
    dynamic.animation.save(
        "outfile.gif",
        fps=10,
        progress_callback=lambda i, n: bar.update(i),
    )
```
To install progressbar2, use the following command in a terminal.
```
pip install progressbar2
```
To view a full description of each class, take a look at the [full documentation](https://davifeliciano.github.io/ising_animate/index.html).
