"""Render individual simulation grids and mosaics from saved spin data."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba

from spin_system import FileManager


class Plotter:
    """Create and save plots for configured simulation data.

    Args:
        args: Parsed config and display options.
        selected_sim_names: Optional ordered
            simulation names or numeric IDs.
        file_manager: Manager for loading data and resolving paths.
        data_source: Source name, such as ``grids`` or ``magnitude``.

    Returns:
        Plotter: A plotter instance.

    """

    def __init__(
        self,
        args: argparse.Namespace,
        selected_sim_names: Sequence[str | int | bytes] | None = None,
        file_manager: FileManager | None = None,
        data_source: str | None = None,
    ) -> None:
        """Initialize plot settings and resolve the selected data source.

        Args:
            args: Parsed config and display options.
            selected_sim_names: Optional
                ordered simulation names or numeric IDs.
            file_manager: Manager for loading data and paths.
            data_source: Source name, such as ``grids`` or ``magnitude``.

        Returns:
            None: Initializes the plot settings and selected data source.

        Raise:
            ValueError: If the gap, source selection, or data source is invalid.
        """
        self.config_name = args.config
        self.file_manager = file_manager or FileManager()
        try:
            self.config = self.file_manager.load_config(self.config_name)
        except FileNotFoundError:
            pass
        self.first_sim = args.first_sim
        self.last_sim = args.last_sim
        self.sim_name = args.sim_name if args.sim_name else None
        self.dpi = args.dpi if args.dpi else 300
        self.cmap = args.cmap if args.cmap else "berlin"
        self.show_info = getattr(args, "show_info", True)
        self.show_axis_labels = getattr(args, "show_axis_labels", True)
        self.ratio = args.ratio
        component = getattr(args, "component", None)
        self.components = [component] if isinstance(component, str) else component
        self.gap = getattr(args, "gap", 0)
        if self.gap < 0:
            raise ValueError("gap must be zero or a positive integer")
        self.gap_color = to_rgba(getattr(args, "gap_color", "white"))
        self.selected_sim_names = (
            [
                self._normalize_simulation_name(value)
                for value in selected_sim_names
            ]
            if selected_sim_names is not None
            else None
        )
        self.base_path = self.file_manager.paths.plots_root / self.config_name
        self.data_source = data_source or getattr(args, "data", "grids")
        if isinstance(self.data_source, (list, tuple)):
            if len(self.data_source) != 1:
                raise ValueError("Plotter accepts one data source per instance")
            self.data_source = self.data_source[0]
        if self.data_source not in {"grids", "magnitude"}:
            raise ValueError(f"Unknown data source: {self.data_source}")
        self.data_base_path = (
            self.base_path
            if self.data_source == "grids"
            else self.base_path / self.data_source
        )

    _COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}

    def _component_color_label(self, component: str = "z") -> str:
        """Return the colorbar label for the selected source and component.

        Args:
            component: Spin component represented in the plot.

        Returns:
            str: Human-readable colorbar label.

        """
        return {
            "grids": r"$S_z$",
            "magnitude": f"Log magnitude ({component} component)",
        }[self.data_source]

    def _color_limits(self, values: np.ndarray) -> tuple[float, float]:
        """Calculate color limits for one data array.

        Args:
            values: Values used to derive the color limits.

        Returns:
            tuple[float, float]: Minimum and maximum color limits.

        """
        if self.data_source == "grids":
            return -1.0, 1.0
        finite_values = np.asarray(values)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return -1.0, 1.0
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if vmin == vmax:
            vmax = vmin + 1e-8
        return vmin, vmax

    def _shared_color_limits(self, sim_names: list[str]) -> tuple[float, float]:
        """Calculate common color limits over the selected simulations.

        Args:
            sim_names: Simulation identifiers whose arrays are inspected.

        Returns:
            tuple[float, float]: Minimum and maximum color limits.

        """
        if self.data_source == "grids" or not sim_names:
            return -1.0, 1.0
        limits = [
            self._color_limits(
                self.file_manager.load_array(self.config_name, name, source=self.data_source)
            )
            for name in sim_names
        ]
        return min(low for low, _ in limits), max(high for _, high in limits)

    @staticmethod
    def _normalize_simulation_name(simulation: str | int | bytes) -> str:
        """Normalize a simulation identifier to the project's naming format.

        Args:
            simulation: Numeric ID or simulation name.

        Returns:
            str: A name in ``sim_XXXXXX`` format when the input is numeric.

        """
        sim_name = (
            simulation.decode() if isinstance(simulation, bytes) else str(simulation)
        ).strip()
        if sim_name.isdecimal():
            return f"sim_{int(sim_name):06d}"
        return sim_name

    @staticmethod
    def _create_parameter_text(
        hamiltonian_terms: dict[str, bool],
        parameter_set: dict[str, Any],
    ) -> str:
        """Format active Hamiltonian parameters for a figure annotation.

        Args:
            hamiltonian_terms: Interaction names and enabled flags.
            parameter_set: Values for the selected simulation.

        Returns:
            str: Formatted enabled interaction parameters.

        """
        fig_text = ""
        energy_contributions = {
            "use_heisenberg": lambda: r"$\bf{J}$"+f"={parameter_set['J']:.2f}, ",
            "use_dmi": lambda: r"$\beta$"+f"={parameter_set['DMI']:.2f}, ",
            "use_dipole": lambda: r"$\bf{D}$"+f"={parameter_set['D']:.2f}, ",
            "use_ani": lambda: r"$\bf{K}$"+f"={parameter_set['K']:.2f}, ",
            }

        for term, use in hamiltonian_terms.items():
            if use and term in energy_contributions:
                fig_text += f"{ energy_contributions[term]()}"

        return fig_text

    def create_sim_name_list(self) -> list[str]:
        """Select simulation names from explicit names or an inclusive range.

        Args:
            None: Selection settings are read from this plotter instance.

        Returns:
            list[str]: Ordered simulation names to plot.

        Raise:
            ValueError: If neither an explicit name nor a valid range is available.
        """
        if self.selected_sim_names is not None:
            return list(self.selected_sim_names)
        if self.sim_name is not None:
            return [self.sim_name]
        elif self.first_sim and self.last_sim:
            if self.last_sim > max(self.config['parameter_sets'].keys()):
                end_idx = int(max(self.config['parameter_sets'].keys()).split("_")[-1])
            else:
                end_idx = int(self.last_sim.split("_")[-1])
            start_idx = int(self.first_sim.split("_")[-1])
            return [f"sim_{i:06d}" for i in range(start_idx, end_idx + 1)]
        else:
            raise ValueError(
                "Provide either sim_name or both first_sim and last_sim."
            )


    def plot_imshow(self) -> None:
        """Create and save component-wise image plots for selected simulations.

        Args:
            None: Plot settings and selection come from this instance.

        Returns:
            None: Saves the selected component images.

        Raise:
            ValueError: If magnitude data is selected without spin components.
            OSError: If an image cannot be saved.
        """

        if self.data_source == "magnitude":
            if not self.components:
                raise ValueError("component (x, y or z) is required for imshow with magnitude data")
            components = list(dict.fromkeys(self.components))
        else:
            components = ["z"]

        sim_names = self.create_sim_name_list()
        vmin, vmax = self._shared_color_limits(sim_names)
        for sim_name in sim_names:
            parameter_set = self.config['parameter_sets'][sim_name]
            spins = self.file_manager.load_array(
                self.config_name,
                sim_name,
                source=self.data_source,
            )

            for component in components:
                values = spins[:, :, self._COMPONENT_INDEX[component]]
                suffix = f"_{component}" if self.data_source == "magnitude" else ""

                fig, ax = plt.subplots(figsize=(6,6))

                im = ax.imshow(
                    values,
                    cmap=self.cmap,
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                )
                if not self.show_axis_labels:
                    ax.tick_params(
                        axis="both",
                        which="both",
                        labelbottom=False,
                        labelleft=False,
                    )

                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label(self._component_color_label(component))

                if self.show_info:
                    fig_text = self._create_parameter_text(
                        self.config['hamiltonian_terms'], parameter_set
                    )
                    plt.figtext(0.5, 0.05, fig_text, ha="center", fontsize=10, wrap=True)
                    ax.set_title(f"{self.config_name}/{sim_name}")

                plt.tight_layout()
                output_dir = self.data_base_path / "imshow"
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / f"{sim_name}{suffix}.png", dpi=300, bbox_inches='tight')
                plt.close()


    def plot_quiver(self) -> None:
        """Create and save vector-field plots for selected simulations.

        Args:
            None: Plot settings and selection come from this instance.

        Returns:
            None: Saves a vector-field image for each selected simulation.

        """
        sim_names = self.create_sim_name_list()
        for sim_name in sim_names:
            spins = self.file_manager.load_array(
                self.config_name,
                sim_name,
                source=self.data_source,
            )

            L = spins.shape[0]

            Sx = spins[:, :, 0]
            Sy = spins[:, :, 1]
            Sz = spins[:, :, 2]

            X, Y = np.meshgrid(np.arange(L), np.arange(L))

            vmin, vmax = self._color_limits(Sz)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.quiver(
                X, Y, Sx, Sy, Sz,
                scale=22,
                cmap=self.cmap,
                norm=norm,
                width=0.01,
                pivot='mid',
                minlength=0.1,
                headwidth=1.5,
            )

            cbar = fig.colorbar(ax.collections[0], ax=ax, shrink=0.8)
            cbar.set_label(self._component_color_label())

            ax.set_aspect('equal')
            ax.set_xlim(-1, L)
            ax.set_ylim(-1, L)
            if self.show_axis_labels:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            else:
                ax.tick_params(
                    axis="both",
                    which="both",
                    labelbottom=False,
                    labelleft=False,
                )

            ax.axis('on')

            output_dir = self.data_base_path / "quiver"
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f"{sim_name}.png", dpi=300, bbox_inches='tight')
            plt.close()


    def _mosaic_sim_names(self, folder_path: Path) -> list[str]:
        """Return the simulation names available for mosaic plotting.

        Args:
            folder_path: Directory containing the simulation arrays.

        Returns:
            list[str]: Ordered simulation names to plot.

        """
        if self.selected_sim_names is not None:
            return self.create_sim_name_list()
        return sorted(
            file.stem
            for file in folder_path.iterdir()
            if file.is_file()
            and file.suffix == ".npy"
            and self.first_sim <= file.stem <= self.last_sim
        )

    def _load_mosaic_arrays(
        self,
        sim_names: list[str],
        *,
        vector_field: bool,
    ) -> list[np.ndarray]:
        """Load and extract scalar or vector data for a mosaic.

        Args:
            sim_names: Simulation identifiers to load.
            vector_field: Whether to retain the first three spin components.

        Returns:
            list[np.ndarray]: Loaded arrays accepted for mosaic plotting.

        """
        arrays: list[np.ndarray] = []
        for sim_name in sim_names:
            try:
                arr = self.file_manager.load_array(
                    self.config_name,
                    sim_name,
                    source=self.data_source,
                )
                if vector_field:
                    if arr.ndim >= 3 and arr.shape[-1] >= 3:
                        arrays.append(arr)
                    else:
                        print(
                            f"Warning: Simulation {sim_name} does not have enough "
                            f"spin components: {arr.shape}"
                        )
                elif arr.ndim >= 2:
                    arrays.append(arr[..., -1])
                else:
                    print(
                        f"Warning: Simulation {sim_name} has too few "
                        f"dimensions: {arr.shape}"
                    )
            except Exception as error:
                print(f"Error processing {sim_name}: {error}")
        return arrays

    def _standardize_mosaic_arrays(
        self,
        arrays: list[np.ndarray],
        *,
        vector_field: bool,
    ) -> tuple[list[np.ndarray], int, int]:
        """Pad arrays to the largest spatial dimensions.

        Args:
            arrays: Scalar or vector arrays to standardize.
            vector_field: Whether arrays contain spin components.

        Returns:
            tuple[list[np.ndarray], int, int]: Standardized arrays, height, and width.

        """
        shapes = np.array([arr.shape[:2] for arr in arrays])
        max_height = int(np.max(shapes[:, 0]))
        max_width = int(np.max(shapes[:, 1]))
        standardized_arrays: list[np.ndarray] = []

        for arr in arrays:
            height, width = arr.shape[:2]
            if (height, width) == (max_height, max_width):
                standardized_arrays.append(arr)
                continue

            if vector_field:
                standardized = np.full((max_height, max_width, 3), np.nan)
                standardized[:height, :width, :] = arr[:max_height, :max_width, :3]
            else:
                standardized = np.full((max_height, max_width), np.nan)
                standardized[:height, :width] = arr[:max_height, :max_width]
            standardized_arrays.append(standardized)

        return standardized_arrays, max_height, max_width

    def _mosaic_layout(
        self,
        n_arrays: int,
        max_height: int,
        max_width: int,
    ) -> tuple[int, int, int, int]:
        """Choose the mosaic grid that best matches the configured aspect ratio.

        Args:
            n_arrays: Number of simulation arrays.
            max_height: Height of each standardized tile.
            max_width: Width of each standardized tile.

        Returns:
            tuple[int, int, int, int]: Columns, rows, mosaic height, and width.

        """
        target_ratio = self.ratio[0] / self.ratio[1]
        best_ratio_diff = float("inf")
        best_cols = int(np.ceil(np.sqrt(n_arrays)))

        for test_cols in range(1, n_arrays + 1):
            test_rows = int(np.ceil(n_arrays / test_cols))
            width = test_cols * max_width + self.gap * (test_cols - 1)
            height = test_rows * max_height + self.gap * (test_rows - 1)
            ratio_diff = abs(width / height - target_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_cols = test_cols

        cols = best_cols
        rows = int(np.ceil(n_arrays / cols))
        mosaic_height = rows * max_height + self.gap * (rows - 1)
        mosaic_width = cols * max_width + self.gap * (cols - 1)
        return cols, rows, mosaic_height, mosaic_width

    @staticmethod
    def _mosaic_figure_size(
        mosaic_height: int,
        mosaic_width: int,
    ) -> tuple[float, float]:
        """Return a figure size that preserves square mosaic tiles.

        Args:
            mosaic_height: Mosaic height in pixels.
            mosaic_width: Mosaic width in pixels.

        Returns:
            tuple[float, float]: Figure width and height in inches.

        """
        base_width = 20
        return base_width, base_width / (mosaic_width / mosaic_height)

    def _mosaic_output_path(
        self,
        output_name: str,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Resolve the output path for a mosaic image.

        Args:
            output_name: Name of the mosaic image.
            output_dir: Optional output directory override.

        Returns:
            Path: Destination path for the mosaic.

        """
        if output_dir is not None:
            return Path(output_dir) / output_name
        output_root = self.file_manager.paths.mosaic_root
        if self.data_source != "grids":
            output_root /= self.data_source
        return output_root / output_name


    def plot_mosaic(
        self,
        output_name: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Create a mosaic image from the selected simulation arrays.

        Args:
            output_name: Optional name for the output image.
            output_dir: Optional output directory.

        Returns:
            None: Saves a mosaic of the selected simulation arrays.

        """

        # Überprüfe, ob der Ordner existiert
        folder_path = self.file_manager.paths.source_directory(
            self.config_name,
            self.data_source,
        )
        if not folder_path.exists():
            print(f"Error: Directory '{folder_path}' does not exist.")
            return

        sim_names = self._mosaic_sim_names(folder_path)

        print(f"Found files: {len(sim_names)}")

        processed_arrays = self._load_mosaic_arrays(
            sim_names,
            vector_field=False,
        )

        if not processed_arrays:
            print("No arrays could be processed.")
            return

        print(f"\nSuccessfully processed arrays: {len(processed_arrays)}")

        standardized_arrays, max_height, max_width = self._standardize_mosaic_arrays(
            processed_arrays,
            vector_field=False,
        )
        cols, rows, mosaic_height, mosaic_width = self._mosaic_layout(
            len(standardized_arrays),
            max_height,
            max_width,
        )

        # Erstelle das große Mosaik-Array
        final_mosaic = np.full((mosaic_height, mosaic_width), np.nan)

        # Füge die standardisierten Arrays in das Mosaik ein
        for idx, array in enumerate(standardized_arrays):
            row = idx // cols
            col = idx % cols

            # Berechne die Position im großen Mosaik
            y_start = row * (max_height + self.gap)
            y_end = y_start + max_height
            x_start = col * (max_width + self.gap)
            x_end = x_start + max_width

            # Füge das Array in das Mosaik ein
            final_mosaic[y_start:y_end, x_start:x_end] = array

        #print(f"Finale Mosaik-Dimension: {final_mosaic.shape}")

        # Erstelle und speichere den Plot
        fig, ax = plt.subplots(
            figsize=self._mosaic_figure_size(mosaic_height, mosaic_width)
        )

        ax.imshow(final_mosaic, cmap=self.cmap, aspect='equal')

        if self.gap:
            gap_overlay = np.zeros((*final_mosaic.shape, 4), dtype=float)
            for row in range(rows - 1):
                y_start = (row + 1) * max_height + row * self.gap
                gap_overlay[y_start:y_start + self.gap, :, :] = self.gap_color
            for col in range(cols - 1):
                x_start = (col + 1) * max_width + col * self.gap
                gap_overlay[:, x_start:x_start + self.gap, :] = self.gap_color
            ax.imshow(
                gap_overlay,
                aspect='equal',
                interpolation='nearest',
                zorder=1,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)

        if output_name is None:
            output_name = f"{self.config_name}_mosaic.png"
        output_path = self._mosaic_output_path(output_name, output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close()

    def plot_mosaic_quiver(self) -> None:
        """Create and save a mosaic of vector-field plots.

        Args:
            None: Plot settings and selection come from this instance.

        Returns:
            None: Saves a mosaic of vector-field plots.

        """

        # Scanne alle .npy Dateien im grids Verzeichnis
        folder_path = self.file_manager.paths.source_directory(
            self.config_name,
            self.data_source,
        )
        if not folder_path.exists():
            print(f"Error: Directory not found: {folder_path}")
            return

        sim_names = self._mosaic_sim_names(folder_path)

        print(f"Found simulations: {len(sim_names)}")

        processed_arrays = self._load_mosaic_arrays(
            sim_names,
            vector_field=True,
        )

        if not processed_arrays:
            print("No arrays could be processed.")
            return

        print(f"\nSuccessfully processed arrays: {len(processed_arrays)}")

        standardized_arrays, max_height, max_width = self._standardize_mosaic_arrays(
            processed_arrays,
            vector_field=True,
        )
        cols, rows, mosaic_height, mosaic_width = self._mosaic_layout(
            len(standardized_arrays),
            max_height,
            max_width,
        )

        fig = plt.figure(figsize=self._mosaic_figure_size(mosaic_height, mosaic_width))
        width_ratios = []
        for col in range(cols):
            width_ratios.append(max_width)
            if self.gap and col < cols - 1:
                width_ratios.append(self.gap)
        height_ratios = []
        for row in range(rows):
            height_ratios.append(max_height)
            if self.gap and row < rows - 1:
                height_ratios.append(self.gap)
        grid = fig.add_gridspec(
            len(height_ratios),
            len(width_ratios),
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=0,
            wspace=0,
        )
        if self.gap:
            for grid_row in range(len(height_ratios)):
                for grid_col in range(len(width_ratios)):
                    if grid_row % 2 == 1 or grid_col % 2 == 1:
                        gap_ax = fig.add_subplot(grid[grid_row, grid_col])
                        gap_ax.set_facecolor(self.gap_color)
                        gap_ax.set_xticks([])
                        gap_ax.set_yticks([])
                        for spine in gap_ax.spines.values():
                            spine.set_visible(False)

        # Globale min/max für z-Komponente (für einheitliche Farbskala)
        all_z_values = []
        for arr in standardized_arrays:
            z_vals = arr[..., 2]
            valid_z = z_vals[~np.isnan(z_vals)]
            if len(valid_z) > 0:
                all_z_values.extend(valid_z.flatten())

        if all_z_values:
            vmin = np.min(all_z_values)
            vmax = np.max(all_z_values)
        else:
            vmin, vmax = -1, 1

        norm = Normalize(vmin=vmin, vmax=vmax)

        # Erstelle Quiver-Plots für jede Simulation
        for idx, arr in enumerate(standardized_arrays):
            row = idx // cols
            col = idx % cols
            grid_row = row * 2 if self.gap else row
            grid_col = col * 2 if self.gap else col
            ax = fig.add_subplot(grid[grid_row, grid_col])

            # Extrahiere die drei Komponenten
            spin_x = arr[..., 0]  # x-Komponente
            spin_y = arr[..., 1]  # y-Komponente
            spin_z = arr[..., 2]  # z-Komponente (für Farbe)

            # Erstelle Koordinatengitter
            h, w = spin_x.shape
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)

            # Erstelle Quiver-Plot
            # U = x-Komponente des Spin-Vektors (Richtung in x)
            # V = y-Komponente des Spin-Vektors (Richtung in y)
            # Farbe basierend auf z-Komponente
            ax.quiver(
                X, Y, spin_x, spin_y,
                spin_z,  # Farbe basierend auf z-Komponente
                cmap="berlin",
                norm=norm,
                angles='xy',
                scale_units='xy',
                scale=1.0,
                pivot='middle',
                headwidth=3,
                headlength=4,
                headaxislength=3
            )

            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(-0.5, h - 0.5)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        output_path = self._mosaic_output_path(
            f"{self.config_name}_mosaic_quiver.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close()

        print(f"Quiver mosaic saved: {output_path}")
