import argparse
import time
from matplotlib.animation import PillowWriter

import ising

# Creating and parsing the cl arguments
parser = argparse.ArgumentParser(
    description="A tool to generate gif animations of the Ising Model."
)

parser.add_argument(
    "-s",
    "--size",
    type=int,
    nargs="?",
    default=128,
    const=128,
    help="the size of the lattice of spins. Default value is 128.",
)

parser.add_argument(
    "-t",
    "--temp",
    type=float,
    nargs="?",
    default=2.0,
    const=2.0,
    help="the temperature of the lattice of spins, in energy units. Default is 2.0.",
)

parser.add_argument(
    "-j",
    "--j-value",
    type=float,
    nargs="+",
    default=[1.0],
    metavar=("J_ROW", "J_COL"),
    help="the coefficient of interaction between neighboring spins on the lattice. "
    "If a pair of values is given, the first will dictate the interaction "
    "between row neighbors, and the second, the interaction between column neighbors. "
    "Default is (1.0, 1.0).",
)

parser.add_argument(
    "-f",
    "--field",
    type=float,
    nargs="?",
    default=0.0,
    const=0.0,
    help="the external magnetic field over the lattice of spins. Default is 0.",
)

parser.add_argument(
    "-i",
    "--init-state",
    type=str,
    nargs="?",
    default="random",
    const="random",
    help="the initial orientation of the spins in the lattice. "
    'Available options are "random", "up" and "down". '
    'Default is "random".',
)

parser.add_argument(
    "--frames",
    type=int,
    nargs="?",
    default=60,
    const=60,
    help="number of frames of the animation. Default is 60.",
)

parser.add_argument(
    "--interval",
    type=int,
    nargs="?",
    default=100,
    const=100,
    help="interval between each frame, in miliseconds. Default is 100.",
)

parser.add_argument(
    "--time-series",
    action="store_true",
    help="include time series of the relevant macroscopic quantities, "
    "which are the mean energy; the mean magnetic moment, i. e. the magnetization; "
    "the specific heat and the magnetic susceptibility.",
)

parser.add_argument(
    "-o",
    "--output",
    type=str,
    nargs="?",
    default="images/ising.gif",
    const="images/ising.gif",
    help='name of the output file. Default is "images/ising.gif".',
)

args = parser.parse_args()

shape = (args.size, args.size)
temp = args.temp

try:
    j_value = args.j_value[0], args.j_value[1]
except IndexError:
    j_value = args.j_value[0]

field = args.field
init_state = args.init_state
time_series = args.time_series
interval = args.interval
frames = args.frames
output = args.output

# Creating an AnimatedIsing instance with the given options
ani_ising = ising.AnimatedIsing(
    shape, temp, j_value, field, init_state, time_series, interval, frames
)


def to_string(seconds: float) -> str:
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    return f"{minutes} minutes and {seconds} seconds"


# Saving animation and computing rendering time
print("Rendering animation...")
start = time.time()
fps = 1000 / interval
ani_ising.animation.save(output, writer=PillowWriter(fps))
total = time.time() - start
print(f"Done in {to_string(total)}! Animation saved as {output}.")
