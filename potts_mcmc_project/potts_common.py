"""Common input checks for the reference and compiled samplers."""

import numpy as np


def validate_inputs(temperatures, size, field, initial_state, burn_sweeps,
                    samples, sample_interval, seed):
    temperatures = np.ascontiguousarray(temperatures, dtype=np.float64)
    if temperatures.ndim != 1 or not len(temperatures):
        raise ValueError("temperatures must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0):
        raise ValueError("temperatures must be finite and positive")
    if np.any(np.diff(temperatures) < 0):
        raise ValueError("temperatures must be in nondecreasing order")
    if not np.isfinite(field):
        raise ValueError("field must be finite")
    values = (size, burn_sweeps, samples, sample_interval, seed)
    if any(not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError("size, sweep counts and seed must be integers")
    ordered = isinstance(initial_state, (int, np.integer)) and initial_state in (0, 1, 2)
    random = isinstance(initial_state, str) and initial_state == 'random'
    if not (ordered or random):
        raise ValueError("initial_state must be 0, 1, 2, or 'random'")
    if size < 2 or burn_sweeps < 0 or samples < 1 or sample_interval < 1:
        raise ValueError("invalid lattice size or sweep counts")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return temperatures
