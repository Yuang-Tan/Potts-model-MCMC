"""Independent checks for both implementations."""

import itertools
import unittest

import numpy as np

import fast_potts
import reference_potts


def exact_observables(size, temperature, field):
    """Enumerate the full lattice, independently of either update kernel."""
    x = np.array([1.0, -0.5, -0.5])
    y = np.array([0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2])
    energies, magnitudes, projections = [], [], []
    for state in itertools.product(range(3), repeat=size * size):
        spins = np.array(state).reshape(size, size)
        matches = np.sum(spins == np.roll(spins, 1, axis=0))
        matches += np.sum(spins == np.roll(spins, 1, axis=1))
        energies.append(-1.5 * matches - field * x[spins].sum())
        magnitudes.append(np.hypot(x[spins].mean(), y[spins].mean()))
        projections.append(x[spins].mean())
    energies = np.array(energies)
    weights = np.exp(-(energies - energies.min()) / temperature)
    weights /= weights.sum()
    return weights @ magnitudes, weights @ projections, weights @ energies / size**2


class PottsTests(unittest.TestCase):
    def test_both_samplers_against_exact_enumeration(self):
        # Statistical comparison: the backends do not share a random stream.
        temperature, field = 1.8, 0.3
        expected = exact_observables(2, temperature, field)
        for backend, start in itertools.product((reference_potts, fast_potts), (0, 'random')):
            with self.subTest(backend=backend.__name__, initial_state=start):
                result = backend.simulate([temperature], size=2, field=field,
                                          initial_state=start,
                                          burn_sweeps=2000, samples=30000,
                                          sample_interval=1, seed=743)
                for name, exact in zip(('magnetisation', 'mx', 'energy'), expected):
                    self.assertAlmostEqual(float(result[name].mean()), exact, delta=0.035)

    def test_three_by_three_against_exact_with_replicate_uncertainty(self):
        temperature, field = 1.8, 0.3
        exact = exact_observables(3, temperature, field)
        for backend in (reference_potts, fast_potts):
            with self.subTest(backend=backend.__name__):
                estimates = []
                for seed in range(6):
                    result = backend.simulate([temperature], size=3, field=field,
                                              burn_sweeps=1000, samples=5000,
                                              sample_interval=1, seed=1000 + seed)
                    estimates.append([result[name].mean() for name in ('magnetisation', 'mx', 'energy')])
                estimates = np.array(estimates)
                mean = estimates.mean(axis=0)
                sem = estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))
                # Independent-chain uncertainty accounts for serial correlation.
                # The small tolerance floor avoids overconfidence in six replicates.
                self.assertTrue(np.all(np.abs(mean - exact) < 5 * sem + 0.005),
                                msg=f"{backend.__name__}: {mean}, exact={exact}, SEM={sem}")

    def test_snapshot_recomputation_and_reproducibility(self):
        args = dict(field=0.2, burn_sweeps=0,
                    samples=12, sample_interval=3, seed=17)
        x = np.array([1.0, -0.5, -0.5])
        y = np.array([0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2])
        for backend, start, size in itertools.product((reference_potts, fast_potts), (2, 'random'), (2, 6)):
            with self.subTest(backend=backend.__name__, initial_state=start, size=size):
                first = backend.simulate([0.8, 1.5, 2.5], size=size, initial_state=start, **args)
                second = backend.simulate([0.8, 1.5, 2.5], size=size, initial_state=start, **args)
                for key in first:
                    np.testing.assert_array_equal(first[key], second[key])
                for i, spins in enumerate(first['snapshots']):
                    mx = x[spins].mean()
                    magnitude = np.hypot(mx, y[spins].mean())
                    matches = np.sum(spins == np.roll(spins, 1, axis=0))
                    matches += np.sum(spins == np.roll(spins, 1, axis=1))
                    energy = (-1.5 * matches - args['field'] * x[spins].sum()) / spins.size
                    self.assertAlmostEqual(first['mx'][i, -1], mx)
                    self.assertAlmostEqual(first['magnetisation'][i, -1], magnitude)
                    self.assertAlmostEqual(first['energy'][i, -1], energy)

    def test_random_start_is_not_an_ordered_state(self):
        # One cold sweep cannot create disorder from an ordered start. A truly
        # random initial lattice, however, remains mixed over this short time.
        for backend in (reference_potts, fast_potts):
            with self.subTest(backend=backend.__name__):
                result = backend.simulate([0.001], size=16, initial_state='random',
                                          burn_sweeps=0, samples=1, sample_interval=1, seed=5)
                self.assertLess(result['magnetisation'][0, 0], 0.5)
                self.assertEqual(len(np.unique(result['snapshots'][0])), 3)

    def test_invalid_inputs(self):
        for backend in (reference_potts, fast_potts):
            with self.subTest(backend=backend.__name__):
                for temperatures, kwargs in [([2.0, 1.0], {}), ([0.0], {}),
                                             ([1.0], {'size': 1}), ([1.0], {'initial_state': -1}),
                                             ([1.0], {'initial_state': 'disordered'})]:
                    with self.assertRaises(ValueError):
                        backend.simulate(temperatures, **kwargs)


if __name__ == '__main__':
    unittest.main()
