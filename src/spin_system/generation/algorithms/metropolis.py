"""Metropolis algorithm for spin system simulations. Optimized with Numba.
Therefore, it uses simplified and optimized calculations for spin interactions."""

import math
from typing import Any, NamedTuple

import numpy as np
from numba import jit

from ..utils.prepare_dipole import DipolePreparation


class _SimulationParameters(NamedTuple):
    """Scalar settings shared by the compiled simulation kernels.

    Args:
        L: Linear size of the square spin lattice.
        n_sweeps: Monte Carlo sweeps per temperature.
        use_heisenberg: Whether Heisenberg interactions are enabled.
        J: Heisenberg interaction strength.
        use_dmi: Whether Dzyaloshinskii-Moriya interactions are enabled.
        DMI: Dzyaloshinskii-Moriya interaction strength.
        use_dipole: Whether dipole interactions are enabled.
        D: Dipole interaction strength.
        use_ani: Whether uniaxial anisotropy is enabled.
        K: Anisotropy strength.

    Returns:
        _SimulationParameters: Immutable settings for Numba-compiled functions.
    """

    L: int
    n_sweeps: int
    use_heisenberg: bool
    J: float
    use_dmi: bool
    DMI: float
    use_dipole: bool
    D: float
    use_ani: bool
    K: float


@jit(nopython=True, cache=True, fastmath=True)
def _initialize_local_field(
    spins: np.ndarray,
    parameters: _SimulationParameters,
    dipole_preparation: DipolePreparation | None,
) -> np.ndarray:
    """Build the linear part of the local energy for every lattice site.

    For a proposed spin change, all interaction terms are linear in the
    changed spin.  The field stores their coefficients, so evaluating a
    proposal does not need to scan the dipole neighbours.  The field is
    updated only after an accepted proposal.

    Args:
        spins: Current lattice spins with shape ``(L, L, 3)``.
        parameters: Interaction and lattice settings.
        dipole_preparation: Prepared dipole-neighbour
            arrays, or ``None``.

    Returns:
        np.ndarray: Local interaction fields with shape ``(L, L, 3)``.
    """
    L = parameters.L
    use_heisenberg = parameters.use_heisenberg
    J = parameters.J
    use_dmi = parameters.use_dmi
    DMI = parameters.DMI
    use_dipole = parameters.use_dipole
    D = parameters.D

    local_field = np.zeros((L, L, 3), dtype=np.float64)

    for i in range(L):
        ni = 0 if i == L - 1 else i + 1
        n_i = L - 1 if i == 0 else i - 1

        for j in range(L):
            nj = 0 if j == L - 1 else j + 1
            n_j = L - 1 if j == 0 else j - 1

            field0 = 0.0
            field1 = 0.0
            field2 = 0.0

            if use_heisenberg and J != 0.0:
                field0 -= J * (
                    spins[ni, j, 0] + spins[n_i, j, 0]
                    + spins[i, nj, 0] + spins[i, n_j, 0]
                )
                field1 -= J * (
                    spins[ni, j, 1] + spins[n_i, j, 1]
                    + spins[i, nj, 1] + spins[i, n_j, 1]
                )
                field2 -= J * (
                    spins[ni, j, 2] + spins[n_i, j, 2]
                    + spins[i, nj, 2] + spins[i, n_j, 2]
                )

            if use_dmi and DMI != 0.0:
                right0 = spins[ni, j, 0]
                right2 = spins[ni, j, 2]
                left0 = spins[n_i, j, 0]
                left2 = spins[n_i, j, 2]
                up1 = spins[i, nj, 1]
                up2 = spins[i, nj, 2]
                down1 = spins[i, n_j, 1]
                down2 = spins[i, n_j, 2]

                # DMI contribution to the local interaction field.
                field0 += DMI * (right2 - left2)
                field1 += DMI * (down2 - up2)
                field2 += DMI * (left0 - right0 + up1 - down1)

            if use_dipole and D != 0.0:
                if dipole_preparation is not None:
                    offs_arr = dipole_preparation[0]
                    uxs_arr = dipole_preparation[1]
                    uys_arr = dipole_preparation[2]
                    invr3_arr = dipole_preparation[3]

                    for idx in range(offs_arr.shape[0]):
                        dx = offs_arr[idx, 0]
                        dy = offs_arr[idx, 1]

                        dipole_i = i + dx
                        if dipole_i >= L:
                            dipole_i -= L
                        elif dipole_i < 0:
                            dipole_i += L

                        dipole_j = j + dy
                        if dipole_j >= L:
                            dipole_j -= L
                        elif dipole_j < 0:
                            dipole_j += L

                        Sj0 = spins[dipole_i, dipole_j, 0]
                        Sj1 = spins[dipole_i, dipole_j, 1]
                        Sj2 = spins[dipole_i, dipole_j, 2]

                        uxs = uxs_arr[idx]
                        uys = uys_arr[idx]
                        invr3 = invr3_arr[idx]
                        norm_xy_sq = uxs * uxs + uys * uys
                        dot_Sj_r = Sj0 * uxs + Sj1 * uys

                        field0 -= D * (norm_xy_sq * Sj0 - 3.0 * uxs * dot_Sj_r) * invr3
                        field1 -= D * (norm_xy_sq * Sj1 - 3.0 * uys * dot_Sj_r) * invr3
                        field2 -= D * (norm_xy_sq * Sj2) * invr3

            local_field[i, j, 0] = field0
            local_field[i, j, 1] = field1
            local_field[i, j, 2] = field2

    return local_field


@jit(nopython=True, cache=True, fastmath=True)
def _update_local_field(
    local_field: np.ndarray,
    i: int,
    j: int,
    delta0: float,
    delta1: float,
    delta2: float,
    parameters: _SimulationParameters,
    dipole_preparation: DipolePreparation | None,
) -> None:
    """Update neighbouring fields after an accepted spin change.

    Args:
        local_field: Local interaction fields to update in place.
        i: First coordinate of the changed spin.
        j: Second coordinate of the changed spin.
        delta0: Change in the spin's x component.
        delta1: Change in the spin's y component.
        delta2: Change in the spin's z component.
        parameters: Interaction and lattice settings.
        dipole_preparation: Prepared dipole-neighbour
            arrays, or ``None``.

    Returns:
        None: The local field is modified in place.
    """
    L = parameters.L
    use_heisenberg = parameters.use_heisenberg
    J = parameters.J
    use_dmi = parameters.use_dmi
    DMI = parameters.DMI
    use_dipole = parameters.use_dipole
    D = parameters.D
    ni = 0 if i == L - 1 else i + 1
    n_i = L - 1 if i == 0 else i - 1
    nj = 0 if j == L - 1 else j + 1
    n_j = L - 1 if j == 0 else j - 1

    if use_heisenberg and J != 0.0:
        field0 = -J * delta0
        field1 = -J * delta1
        field2 = -J * delta2

        local_field[ni, j, 0] += field0
        local_field[ni, j, 1] += field1
        local_field[ni, j, 2] += field2
        local_field[n_i, j, 0] += field0
        local_field[n_i, j, 1] += field1
        local_field[n_i, j, 2] += field2
        local_field[i, nj, 0] += field0
        local_field[i, nj, 1] += field1
        local_field[i, nj, 2] += field2
        local_field[i, n_j, 0] += field0
        local_field[i, n_j, 1] += field1
        local_field[i, n_j, 2] += field2

    if use_dmi and DMI != 0.0:
        # The changed spin is the source of the +x/+y bonds and the
        # destination of the -x/-y bonds.
        local_field[ni, j, 0] -= DMI * delta2
        local_field[ni, j, 2] += DMI * delta0
        local_field[n_i, j, 0] += DMI * delta2
        local_field[n_i, j, 2] -= DMI * delta0
        local_field[i, nj, 1] += DMI * delta2
        local_field[i, nj, 2] -= DMI * delta1
        local_field[i, n_j, 1] -= DMI * delta2
        local_field[i, n_j, 2] += DMI * delta1

    if use_dipole and D != 0.0:
        if dipole_preparation is None:
            return

        offs_arr = dipole_preparation[0]
        uxs_arr = dipole_preparation[1]
        uys_arr = dipole_preparation[2]
        invr3_arr = dipole_preparation[3]

        for idx in range(offs_arr.shape[0]):
            dx = offs_arr[idx, 0]
            dy = offs_arr[idx, 1]

            dipole_i = i + dx
            if dipole_i >= L:
                dipole_i -= L
            elif dipole_i < 0:
                dipole_i += L

            dipole_j = j + dy
            if dipole_j >= L:
                dipole_j -= L
            elif dipole_j < 0:
                dipole_j += L

            uxs = uxs_arr[idx]
            uys = uys_arr[idx]
            invr3 = invr3_arr[idx]
            norm_xy_sq = uxs * uxs + uys * uys
            dot_delta_r = delta0 * uxs + delta1 * uys

            local_field[dipole_i, dipole_j, 0] -= D * (
                norm_xy_sq * delta0 - 3.0 * uxs * dot_delta_r
            ) * invr3
            local_field[dipole_i, dipole_j, 1] -= D * (
                norm_xy_sq * delta1 - 3.0 * uys * dot_delta_r
            ) * invr3
            local_field[dipole_i, dipole_j, 2] -= D * (
                norm_xy_sq * delta2
            ) * invr3


@jit(nopython=True, cache=True, fastmath=True)
def _monte_carlo_sweep_core_with_field(
    spins: np.ndarray,
    T: float,
    local_field: np.ndarray,
    parameters: _SimulationParameters,
    dipole_preparation: DipolePreparation | None,
) -> np.ndarray:
    """Run Monte Carlo sweeps while maintaining the local interaction field.

    Args:
        spins: Mutable spins with shape ``(L, L, 3)``.
        T: Temperature used for Metropolis acceptance.
        local_field: Current local interaction fields.
        parameters: Interaction and lattice settings.
        dipole_preparation: Prepared dipole-neighbour
            arrays, or ``None``.

    Returns:
        np.ndarray: The updated spin array.
    """
    L = parameters.L
    n_sweeps = parameters.n_sweeps
    use_ani = parameters.use_ani
    K = parameters.K

    beta = 1.0 / T if T > 0 else 0.0
    two_pi = 2.0 * math.pi

    for _ in range(n_sweeps):
        for i in range(L):
            for j in range(L):
                old0 = spins[i, j, 0]
                old1 = spins[i, j, 1]
                old2 = spins[i, j, 2]

                field0 = local_field[i, j, 0]
                field1 = local_field[i, j, 1]
                field2 = local_field[i, j, 2]

                e_old = old0 * field0 + old1 * field1 + old2 * field2
                if use_ani and K != 0.0:
                    e_old -= K * old2 * old2

                cos_theta = 1.0 - 2.0 * np.random.rand()
                new_phi = two_pi * np.random.rand()
                sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
                new0 = sin_theta * math.cos(new_phi)
                new1 = sin_theta * math.sin(new_phi)
                new2 = cos_theta

                e_new = new0 * field0 + new1 * field1 + new2 * field2
                if use_ani and K != 0.0:
                    e_new -= K * new2 * new2

                dE = e_new - e_old

                if dE <= 0.0 or (beta != 0.0 and np.random.random() < math.exp(-dE * beta)):
                    spins[i, j, 0] = new0
                    spins[i, j, 1] = new1
                    spins[i, j, 2] = new2

                    # Use the values after the float32 assignment.  This
                    # keeps the incremental field aligned with ``spins``.
                    delta0 = spins[i, j, 0] - old0
                    delta1 = spins[i, j, 1] - old1
                    delta2 = spins[i, j, 2] - old2

                    _update_local_field(
                        local_field, i, j, delta0, delta1, delta2,
                        parameters, dipole_preparation
                    )

    return spins


@jit(nopython=True, cache=True, fastmath=True)
def run_annealing_core(
    spins: np.ndarray,
    temps: np.ndarray,
    parameters: _SimulationParameters,
    dipole_preparation: DipolePreparation | None,
    random_seed: int,
) -> np.ndarray:
    """Run all annealing temperatures inside the Numba-compiled kernel.

    Args:
        spins: Initial lattice spins with shape ``(L, L, 3)``.
        temps: One-dimensional array of annealing temperatures.
        parameters: Interaction and lattice settings.
        dipole_preparation: Prepared dipole-neighbour
            arrays, or ``None``.
        random_seed: Non-negative seed, or a negative value to retain RNG state.

    Returns:
        np.ndarray: The final spin array.
    """
    if random_seed >= 0:
        np.random.seed(random_seed)

    local_field = _initialize_local_field(spins, parameters, dipole_preparation)

    for t_idx in range(len(temps)):
        spins = _monte_carlo_sweep_core_with_field(
            spins, temps[t_idx], local_field, parameters, dipole_preparation
        )
    return spins


def monte_carlo_sweep_metropolis(
    config: dict[str, Any],
    sim_name: str,
    spins: np.ndarray,
    temps: np.ndarray,
    dipole_preparation: DipolePreparation | None,
    seed: int | None = None,
) -> np.ndarray:
    """Run a complete temperature schedule using the Numba annealing kernel.

    Args:
        config: Simulation configuration with Hamiltonian flags
            and parameters.
        sim_name: Key identifying the selected parameter set.
        spins: Initial spin array with shape ``(L, L, 3)``.
        temps: One-dimensional array of annealing temperatures.
        dipole_preparation: Prepared dipole-neighbour
            arrays, or ``None``.
        seed: Optional random seed for the compiled simulation.

    Returns:
        np.ndarray: The final spin array.
    """

    L = spins.shape[0]
    n_sweeps = int(config['base_parameters']['mc_steps_factor'])

    # Extract flags and parameters once to avoid dict lookups inside the sweep
    parameter_set = config['parameter_sets'][sim_name]
    ham = config['hamiltonian_terms']

    use_heisenberg = ham.get('use_heisenberg', False)
    use_dmi = ham.get('use_dmi', False)
    use_dipole = ham.get('use_dipole', False)
    use_ani = ham.get('use_ani', False)

    J = parameter_set.get('J', 0.0)
    DMI = parameter_set.get('DMI', 0.0)
    D = parameter_set.get('D', 0.0)
    K = parameter_set.get('K', 0.0)

    parameters = _SimulationParameters(
        L, n_sweeps, use_heisenberg, J, use_dmi, DMI, use_dipole, D,
        use_ani, K,
    )
    random_seed = -1 if seed is None else int(seed)
    spins = run_annealing_core(
        spins, temps, parameters, dipole_preparation, random_seed
    )

    return spins
