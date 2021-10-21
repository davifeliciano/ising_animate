"""
This is a module with an example of usage of the class DynamicAnimatedIsing.
An animation of an Ising Model with temperature T(t) = 1.0 + 0.3t
and external magnetic field B(t) = sin(t).
"""
from ..ising import DynamicAnimatedIsing
from math import sin
import progressbar
import arrow

if __name__ == "__main__":
    frames = 100
    interval = 100
    fps = 1000 / interval

    dynamic = DynamicAnimatedIsing(
        shape=(256, 256),  # the shape of the lattice
        temp=lambda t: 1.0 + 0.3 * t,  # temperature as a function of time
        field=lambda t: sin(t),  # external magnetic field as a function of time
        time_series=True,  # plot evolution of physical quantities over time
        interval=100,  # interval of each frame
        frames=frames,  # amount of frames in the animation
    )

    time_string = arrow.now().format("YYYY-MM-DD_HH-mm")
    fig_outfile = f"dynamic_{time_string}.gif"
    print("Rendering animation...")

    with progressbar.ProgressBar(max_value=frames) as bar:
        dynamic.animation.save(
            fig_outfile,
            fps=fps,
            progress_callback=lambda i, n: bar.update(i),
        )

    print(f"Figure saved as {fig_outfile}")
