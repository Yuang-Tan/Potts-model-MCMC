"""Readable Python reference sampler for the three-state Potts model.

The fast implementation in fast_potts.py uses the same physical rules.
Energies and temperatures are in units where J = k_B = 1.
"""

import numpy as np

from potts_common import validate_inputs

# Each state is represented by a unit vector at a separation of 120 degrees.
SPINS = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])


def simulate(temperatures, size=24, field=0.0, initial_state=0, burn_sweeps=1000,
             samples=500, sample_interval=5, seed=0):
    """Heat one chain; retain the final configuration at each temperature.

    One sweep consists of ``size**2`` random-site update attempts. Observations
    are recorded every ``sample_interval`` sweeps after ``burn_sweeps``. The
    arrays ``magnetisation``, ``mx`` and ``energy`` have shape (temperature,
    samples); ``snapshots`` contains the last lattice at each temperature.
    Set ``initial_state='random'`` for independent uniform labels at each site;
    integer states 0, 1 and 2 start fully ordered.
    """
    temperatures = validate_inputs(temperatures, size, field, initial_state,
                                   burn_sweeps, samples, sample_interval, seed)
    rng = np.random.default_rng(seed)
    n = size * size
    if initial_state == 'random':
        lattice = rng.integers(3, size=(size, size), dtype=np.int8)
        total_m = SPINS[lattice].sum(axis=(0, 1))
        # Count right/down bonds once, including periodic boundary bonds.
        matches = sum(np.sum(lattice == np.roll(lattice, 1, axis=axis)) for axis in (0, 1))
        total_energy = -1.5 * matches - field * total_m[0]
    else:
        lattice = np.full((size, size), initial_state, dtype=np.int8)
        total_m = n * SPINS[initial_state].copy()
        total_energy = -3.0 * n - field * total_m[0]  # 2N equal bonds
    shape = (len(temperatures), samples)
    result = {name: np.empty(shape) for name in ('magnetisation', 'mx', 'energy')}
    result['acceptance'] = np.empty(len(temperatures))
    result['snapshots'] = np.empty((len(temperatures), size, size), dtype=np.int8)

    for t_index, temperature in enumerate(temperatures):
        accepted = 0
        attempted = 0

        def sweep(measure):
            nonlocal accepted, attempted, total_energy
            for _ in range(n):
                row, col = rng.integers(size, size=2)
                old = lattice[row, col]
                new = (old + rng.integers(1, 3)) % 3  # either alternative
                neighbours = (lattice[(row - 1) % size, col], lattice[(row + 1) % size, col],
                              lattice[row, (col - 1) % size], lattice[row, (col + 1) % size])
                old_matches = sum(s == old for s in neighbours)
                new_matches = sum(s == new for s in neighbours)
                change_m = SPINS[new] - SPINS[old]
                delta_energy = -1.5 * (new_matches - old_matches) - field * change_m[0]
                accept = delta_energy <= 0 or rng.random() < np.exp(-delta_energy / temperature)
                if accept:
                    lattice[row, col] = new
                    total_m[:] += change_m
                    total_energy += delta_energy
                if measure:
                    accepted += accept
                    attempted += 1

        for _ in range(burn_sweeps):
            sweep(False)
        for sample in range(samples):
            for _ in range(sample_interval):
                sweep(True)
            result['magnetisation'][t_index, sample] = np.linalg.norm(total_m) / n
            result['mx'][t_index, sample] = total_m[0] / n
            result['energy'][t_index, sample] = total_energy / n
        result['acceptance'][t_index] = accepted / attempted
        result['snapshots'][t_index] = lattice
    return result
