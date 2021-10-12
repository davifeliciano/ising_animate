import os
import argparse
from matplotlib.animation import PillowWriter
import progressbar
import arrow

from .ising import AnimatedIsing, CoolingAnimatedIsing
from .timer import timer

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
    nargs=2,
    default=[1.0, 1.0],
    metavar=("J_ROW", "J_COL"),
    help="the coefficient of interaction between neighboring spins on the lattice. "
    "The first will dictate the interaction between row neighbors, and the second, "
    "the interaction between column neighbors. Default is (1.0, 1.0).",
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
    "-c",
    "--cooling",
    type=float,
    nargs=3,
    metavar=("INIT_TEMP", "FINAL_TEMP", "COOL_RATE"),
    help="if provided, the temperature of the system will evolve acording to the "
    "function T(t) = [FINAL_TEMP] + ([INIT_TEMP] - [FINAL_TEMP]) * exp(- [COOL_RATE] * t). "
    "Final temperature is not required to be smaller than initial temperature. "
    "This option overwrites the option --temp.",
)

time_string = arrow.now().format("YYYY-MM-DD_HH-mm-ss")

parser.add_argument(
    "-o",
    "--output",
    type=str,
    nargs="?",
    default=f"ising_{time_string}.gif",
    const=f"ising_{time_string}.gif",
    help='name of the output file. Default is "[DATE]_[TIME].gif".',
)

args = parser.parse_args()


@timer
def main():
    shape = (args.size, args.size)
    temp = args.temp
    j_value = args.j_value
    field = args.field
    init_state = args.init_state
    time_series = args.time_series
    interval = args.interval
    frames = args.frames
    output = args.output

    extension = os.path.splitext(output)[1]
    if extension.lower() != ".gif":
        output += ".gif"

    if args.cooling:
        temp, final_temp, cooling_rate = args.cooling
        ani_ising = CoolingAnimatedIsing(
            shape,
            temp,
            final_temp,
            cooling_rate,
            j_value,
            field,
            init_state,
            time_series,
            interval,
            frames,
        )
    else:
        ani_ising = AnimatedIsing(
            shape,
            temp,
            j_value,
            field,
            init_state,
            time_series,
            interval,
            frames,
        )

    # Saving animation and computing rendering time
    print("Rendering animation...")
    fps = 1000 / interval
    with progressbar.ProgressBar(max_value=frames) as bar:
        ani_ising.animation.save(
            output,
            writer=PillowWriter(fps),
            progress_callback=lambda i, n: bar.update(i),
        )
    print(f"Animation saved as {output}")


if __name__ == "__main__":
    main()
