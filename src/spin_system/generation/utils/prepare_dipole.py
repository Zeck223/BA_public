"""Prepare periodic-lattice offsets and weights for dipole interactions."""

import math
from typing import TypeAlias

import numpy as np

DipolePreparation: TypeAlias = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def prepare_dipole_offsets(
    L: int,
    r_cut: float | None,
    grid_constant: float = 1.0,
) -> DipolePreparation:
    """Build neighbour offsets and radial weights within a dipole cutoff.

    Args:
        L: Number of lattice sites along each side of the square grid.
        r_cut: Physical cutoff radius, or ``None`` to use the
            lattice extent.
        grid_constant: Physical spacing between adjacent lattice sites.

    Returns:
        DipolePreparation: Arrays containing integer offsets, x/y direction
        cosines, and inverse fifth-power distance weights.
    """
    if r_cut is None:
        r_cut = L * grid_constant

    # Skaliere den Cutoff-Radius auf Gittereinheiten
    R_gitter = int(math.ceil(r_cut / grid_constant))

    offs = []
    uxs = []
    uys = []
    invr3 = []

    for dx in range(-R_gitter, R_gitter + 1):
        for dy in range(-R_gitter, R_gitter + 1):
            if dx == 0 and dy == 0:
                continue

            dx_phys = dx * grid_constant
            dy_phys = dy * grid_constant

            r2_phys = dx_phys*dx_phys + dy_phys*dy_phys

            if r2_phys > r_cut*r_cut:
                continue

            r_phys = math.sqrt(r2_phys)

            offs.append((dx, dy))
            uxs.append(dx_phys / r_phys)
            uys.append(dy_phys / r_phys)
            invr3.append(1.0 / (r_phys * r_phys * r_phys * r_phys * r_phys))

    # Konvertierung zu numpy arrays
    offs_arr = np.array(offs, dtype=np.int32)
    uxs_arr = np.array(uxs, dtype=np.float64)
    uys_arr = np.array(uys, dtype=np.float64)
    invr3_arr = np.array(invr3, dtype=np.float64)

    return (offs_arr, uxs_arr, uys_arr, invr3_arr)


if __name__ == "__main__":
    L = 10
    r_cut = 3
    grid_constant = 1

    offs_arr, uxs_arr, uys_arr, invr3_arr = prepare_dipole_offsets(L, r_cut, grid_constant)

    print("Offsets (dx, dy):")
    print(offs_arr)
    print("Unit vectors (ux, uy):")
    print(list(zip(uxs_arr, uys_arr)))
    print("Inverse r^5 values:")
    print(invr3_arr)