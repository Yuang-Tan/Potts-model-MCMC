"""Compare ordered and random starts without changing the physical temperature.

The high-temperature runs record every sweep. At h=0, matching-bond fraction
is -e/3, where e is the Potts energy per spin in units J=1. The independent
random reference is a T=infinity baseline, not a finite-temperature equilibrium
sample. Critical random-start runs supplement the earlier ordered-start data.
"""
from pathlib import Path
import json
import time

import numpy as np
from fast_potts import simulate

ROOT = Path(__file__).resolve().parent


def main():
    started = time.perf_counter()
    size, temperature, total_sweeps, discard, repetitions = 128, 2.25, 4000, 2000, 6
    arrays = {}
    for name, state, seed_base in [('ordered', 0, 20262000), ('random', 'random', 20263000)]:
        magnitudes, energies, snapshots = [], [], []
        seeds = np.arange(seed_base, seed_base + repetitions)
        for i, seed in enumerate(seeds):
            result = simulate([temperature], size=size, field=0, initial_state=state,
                              burn_sweeps=0, samples=total_sweeps,
                              sample_interval=1, seed=int(seed))
            magnitudes.append(result['magnetisation'][0])
            energies.append(result['energy'][0])
            snapshots.append(result['snapshots'][0])
            print(f'{name} high-T run {i + 1}/{repetitions}', flush=True)
        arrays[f'{name}_m'] = np.array(magnitudes)
        arrays[f'{name}_energy'] = np.array(energies)
        arrays[f'{name}_snapshots'] = np.array(snapshots)
        arrays[f'{name}_seeds'] = seeds
    tc = 1.5 / np.log(1 + np.sqrt(3))
    critical_seeds = np.arange(20264000, 20264004)
    critical_m, critical_snapshots = [], []
    for i, seed in enumerate(critical_seeds):
        result = simulate([tc], size=size, field=0, initial_state='random',
                          burn_sweeps=8000, samples=2000, sample_interval=1,
                          seed=int(seed))
        critical_m.append(result['magnetisation'][0])
        critical_snapshots.append(result['snapshots'][0])
        print(f'random critical run {i + 1}/{len(critical_seeds)}', flush=True)
    arrays['critical_random_m'] = np.array(critical_m)
    arrays['critical_random_snapshots'] = np.array(critical_snapshots)
    arrays['critical_random_seeds'] = critical_seeds
    arrays['iid_reference'] = np.random.default_rng(20265000).integers(0, 3, (size, size), dtype=np.int8)
    metadata = {'size': size, 'temperature': temperature, 'field': 0,
                'sweeps_recorded': total_sweeps, 'discard_for_summary': discard,
                'high_temperature_repetitions': repetitions,
                'critical_temperature': tc, 'critical_discarded_sweeps': 8000,
                'critical_measured_sweeps': 2000, 'critical_repetitions': 4,
                'iid_reference_seed': 20265000, 'J': 1, 'k_B': 1,
                'periodic_boundaries': True,
                'elapsed_seconds': round(time.perf_counter()-started, 2)}
    target = ROOT / 'results' / 'disorder_diagnostic.npz'
    np.savez_compressed(target, metadata=json.dumps(metadata), **arrays)
    for name in ('ordered', 'random'):
        m = arrays[f'{name}_m'][:, discard:].mean(axis=1)
        e = arrays[f'{name}_energy'][:, discard:].mean(axis=1)
        for quantity, values in [('m', m), ('e', e), ('matching fraction', -e/3)]:
            print(f'{name}: {quantity} = {values.mean():.6f} +/- {values.std(ddof=1)/np.sqrt(len(values)):.6f}')
    cm = arrays['critical_random_m'].mean(axis=1)
    print('critical random mean m:', cm, 'grand mean', cm.mean())
    print('Saved', target, flush=True)


if __name__ == '__main__':
    main()
