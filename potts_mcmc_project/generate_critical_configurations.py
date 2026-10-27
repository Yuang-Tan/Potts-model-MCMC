"""Generate reproducible, large zero-field configurations near the Potts critical point.

The first long-run critical snapshot is used for illustration, without selecting
by appearance. Short- versus long-burn comparisons are only an equilibration
diagnostic: agreement cannot establish equilibration or a fractal dimension.
"""
from pathlib import Path
import numpy as np

from fast_potts import simulate

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results' / 'critical_configurations.npz'


def main():
    size = 128
    tc = 1.5 / np.log(1 + np.sqrt(3))
    short_burn, long_burn, measurement = 2000, 8000, 2000
    n_replicates = 4
    seed_base = 20261101

    # Both start from an ordered state. At zero field, the three ordered states
    # are symmetry equivalent, so use state 0 for all displayed configurations.
    low = simulate([0.75], size=size, field=0, initial_state=0,
                   burn_sweeps=2000, samples=1, sample_interval=1, seed=seed_base)
    high = simulate([2.25], size=size, field=0, initial_state=0,
                    burn_sweeps=2000, samples=1, sample_interval=1, seed=seed_base + 1)

    means = {}
    snapshots = []
    for label, burn, offset in [('short', short_burn, 100), ('long', long_burn, 200)]:
        run_means = []
        for i in range(n_replicates):
            result = simulate([tc], size=size, field=0, initial_state=0,
                              burn_sweeps=burn, samples=measurement,
                              sample_interval=1, seed=seed_base + offset + i)
            run_means.append(result['magnetisation'][0].mean())
            if label == 'long':
                snapshots.append(result['snapshots'][0].copy())
            print(f'{label} burn: run {i + 1}/{n_replicates}, mean m = {run_means[-1]:.4f}', flush=True)
        means[label] = np.array(run_means)

    OUTPUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUTPUT, size=size, critical_temperature=tc, seed_base=seed_base,
                        low_temperature=0.75, high_temperature=2.25,
                        low_high_burn_sweeps=2000, short_burn_sweeps=short_burn,
                        long_burn_sweeps=long_burn, measurement_sweeps=measurement,
                        low_snapshot=low['snapshots'][0], high_snapshot=high['snapshots'][0],
                        critical_snapshots=np.array(snapshots),
                        critical_short_means=means['short'], critical_long_means=means['long'])
    print(f'Saved {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
