"""Metropolis sampling of the 2D three-state Potts model (J = k_B = 1)."""
from ctypes import CDLL, POINTER, c_double, c_int, c_int8, c_uint64
from pathlib import Path
import os
import subprocess

import numpy as np

from potts_common import validate_inputs

_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "potts_core.cpp"
_LIBRARY = _ROOT / "potts_core.so"


def _load_core():
    """Compile the small native kernel when necessary and load it."""
    if not _LIBRARY.exists() or _SOURCE.stat().st_mtime > _LIBRARY.stat().st_mtime:
        compiler = os.environ.get("CXX", "c++")
        try:
            subprocess.run([compiler, "-std=c++17", "-O3", "-shared", "-fPIC",
                            str(_SOURCE), "-o", str(_LIBRARY)], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("A working C++17 compiler is required to build the sampler") from exc
    lib = CDLL(str(_LIBRARY))
    function = lib.potts_run
    function.argtypes = [POINTER(c_double), c_int, c_int, c_double, c_int,
                         c_int, c_int, c_int, c_uint64, POINTER(c_double),
                         POINTER(c_double), POINTER(c_double), POINTER(c_double),
                         POINTER(c_int8)]
    function.restype = None
    return function


def simulate(temperatures, size=24, field=0.0, initial_state=0, burn_sweeps=1000,
             samples=500, sample_interval=5, seed=0):
    """Run one heating chain, retaining its final state between temperatures.

    A sweep consists of ``size**2`` independent random-site proposals. After
    equilibration at each temperature, record ``samples`` observations separated
    by ``sample_interval`` sweeps. Sample magnitudes before averaging: <|m|>.
    The C++ kernel is compiled on the first call. Seeds make independent runs
    reproducible with the same compiler and C++ standard library.
    ``initial_state='random'`` draws independent uniform labels at each site;
    integer states 0, 1 and 2 start fully ordered.
    """
    temperatures = validate_inputs(temperatures, size, field, initial_state,
                                   burn_sweeps, samples, sample_interval, seed)
    native_state = -1 if initial_state == 'random' else initial_state

    shape = (temperatures.size, samples)
    result = {key: np.empty(shape, dtype=np.float64) for key in ("magnetisation", "mx", "energy")}
    result["acceptance"] = np.empty(temperatures.size, dtype=np.float64)
    result["snapshots"] = np.empty((temperatures.size, size, size), dtype=np.int8)
    double_ptr = lambda array: array.ctypes.data_as(POINTER(c_double))
    _load_core()(double_ptr(temperatures), temperatures.size, size, field, native_state,
                 burn_sweeps, samples, sample_interval, seed,
                 double_ptr(result["magnetisation"]), double_ptr(result["mx"]),
                 double_ptr(result["energy"]), double_ptr(result["acceptance"]),
                 result["snapshots"].ctypes.data_as(POINTER(c_int8)))
    return result
