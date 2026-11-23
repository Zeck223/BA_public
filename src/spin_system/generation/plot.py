"""Command-line entry point for plotting generated spin simulation data."""

import argparse
from collections.abc import Sequence

from spin_system import FileManager
from spin_system.core.paths import ProjectPaths

from .utils.cli import CompactHelpFormatter
from .utils.plotter import Plotter


def main(
    arguments: Sequence[str] | None = None,
    paths: ProjectPaths | None = None,
) -> int:
    """Parse plotting options and dispatch the requested plot operation.

    Args:
        arguments: Optional command-line arguments.
        paths: Paths used by the file manager.

    Returns:
        int: Exit status, where zero indicates successful plotting.
    """

    parser = argparse.ArgumentParser(
        description="Plotting tool for simulation data",
        formatter_class=CompactHelpFormatter,
    )

    parser.add_argument(
            "--config",
            "-c",
            help="Name of the configuration",
            required=True
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["imshow", "quiver", "mosaic", "mosaic_quiver"],
        required=True,
        help="Type of plot"
    )

    parser.add_argument(
        "--data",
        "-d",
        nargs="+",
        default=["grids"],
        choices=["grids", "magnitude"],
        metavar="SOURCE",
        help="Data sources to plot: grids, magnitude (default: grids). "
             "Multiple sources are plotted separately.",
    )
    parser.add_argument(
        "--component",
        nargs="+",
        choices=["x", "y", "z"],
        metavar="COMPONENT",
        help="Spin components to plot (x, y, z; several allowed); required for --type imshow with --data magnitude",
    )
    parser.add_argument(
        "--first_sim",
        "-f",
        help="Name of the first simulation",
        default="sim_000001",
    )
    parser.add_argument(
        "--last_sim",
        "-l",
        help="Name of the last simulation",
        default="sim_999999",
    )

    simulation_selection = parser.add_mutually_exclusive_group()
    simulation_selection.add_argument(
        "--sim_name",
        "-s",
        help="Name of one single simulation",
    )
    simulation_selection.add_argument(
        "--sim_names",
        nargs="+",
        metavar="SIM_NAME",
        help="List of simulation names to include, in the given order",
    )

    simulation_selection.add_argument(
        "--sim_ids",
        nargs="+",
        type=int,
        metavar="SIM_ID",
        help="List of numeric simulation IDs (e.g., 1 4 for sim_000001 and sim_000004)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        help="DPI of the plot",
        default=300,
    )

    parser.add_argument(
        "--cmap",
        help="Colormap",
        default="RdBu",
    )
    parser.add_argument(
        "--show-info",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show or hide plot title and Hamiltonian parameters",
    )

    parser.add_argument(
        "--show-axis-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show or hide axis labels and tick numbers",
    )

    parser.add_argument(
        "--ratio",
        type=int,
        nargs=2,
        help="Aspect ratio as two numbers",
        default=[15, 15],
    )

    parser.add_argument(
        "--gap",
        type=int,
        default=0,
        metavar="PIXELS",
        help="Width of mosaic separators in simulation-grid pixels",
    )

    parser.add_argument(
        "--gap_color",
        "--gap-color",
        default="white",
        metavar="COLOR",
        help="Color of the gap between mosaic tiles",
    )

    args = parser.parse_args(arguments)
    if args.gap < 0:
        parser.error("--gap must be zero or a positive integer")
    if args.type == "imshow" and "magnitude" in args.data and args.component is None:
        parser.error("--component {x,y,z} is required for --type imshow with --data magnitude")

    selected_sim_names = args.sim_names if args.sim_names is not None else args.sim_ids
    file_manager = FileManager(paths=paths)
    for data_source in args.data:
        plotter = Plotter(
            args,
            selected_sim_names=selected_sim_names,
            file_manager=file_manager,
            data_source=data_source,
        )

        plot_methods = {
            "imshow": plotter.plot_imshow,
            "quiver": plotter.plot_quiver,
            "mosaic": plotter.plot_mosaic,
            "mosaic_quiver": plotter.plot_mosaic_quiver,
        }
        plot_methods[args.type]()
    return 0


if __name__ == "__main__":
    main()

