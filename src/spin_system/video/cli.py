"""Command-line interface for zooming mosaic videos."""

from __future__ import annotations

import argparse
from pathlib import Path

from spin_system import FileManager
from spin_system.core.cli import CompactHelpFormatter

from .creator import ZoomMosaicVideo


def main(arguments=None, paths=None) -> int:
    """Create a 3:2 MP4 that zooms from a data mosaic to six simulations."""
    parser = argparse.ArgumentParser(
        description="Create a 3:2 mosaic video that zooms in on six simulations",
        formatter_class=CompactHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Configuration name")
    parser.add_argument(
        "--source",
        default="grids",
        help="Data source within the configuration (default: grids)",
    )
    parser.add_argument(
        "--first-sim",
        "--first_sim",
        "-f",
        dest="first_sim",
        help="First simulation to include",
    )
    parser.add_argument(
        "--last-sim",
        "--last_sim",
        "-l",
        dest="last_sim",
        help="Last simulation to include",
    )
    parser.add_argument("--duration", type=float, default=8, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution (6 x 4 inches)")
    parser.add_argument("--cmap", default="berlin", help="Matplotlib colormap")
    parser.add_argument("--output", type=Path, help="Path to the output MP4 file")
    args = parser.parse_args(arguments)

    file_manager = FileManager(paths=paths)
    output = args.output or (
        file_manager.paths.videos_root / args.config / f"{args.config}_zoom.mp4"
    )
    video = ZoomMosaicVideo(
        args.config,
        first_sim=args.first_sim,
        last_sim=args.last_sim,
        source=args.source,
        cmap=args.cmap,
        file_manager=file_manager,
    )
    video.create(output, duration=args.duration, fps=args.fps, dpi=args.dpi)
    print(f"Video saved to: {output}")
    return 0