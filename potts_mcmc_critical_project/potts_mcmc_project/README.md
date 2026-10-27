# Three-state Potts model: Monte Carlo study

This project combines a concise simulation report with a separate analysis of the statistical mechanics behind Monte Carlo sampling. It began as an Honours Statistical Physics assignment.

## Read

- **`MCMC_background.md`** develops the argument from high-dimensional Boltzmann averages to importance sampling, Metropolis sampling and its limitations.
- **`01_potts_exploration.ipynb`** answers the assignment through simulation: energy conventions, sequential heating, initial-state comparisons, fields, error bars, finite size and configurations. Executed outputs and saved data are included.

## Run

Requires Python 3.10 or later. Install the dependencies and open the notebook from this directory:

```bash
python -m pip install -r requirements.txt
jupyter notebook 01_potts_exploration.ipynb
```

The Python sampler and saved-data analysis work without a compiler. The faster sampler and full checks require a C++17 compiler on a Unix-like system. Set `CXX` to choose a compiler; compilation is automatic on first use.

```bash
python -m unittest -v
python experiments.py
python generate_critical_configurations.py
python diagnose_disorder.py
```

These scripts regenerate the corresponding results. Full simulations may take several minutes. Fixed seeds reproduce each implementation on the same software stack; Python and C++ use different random-number generators and are compared statistically.

## Implementation

`reference_potts.py` is the readable Python implementation. `fast_potts.py` calls `potts_core.cpp`; `potts_common.py` validates their shared inputs. Both accept ordered initial states `0`, `1`, `2` or `initial_state='random'`. `test_potts.py` checks exact small-system expectations, snapshot observables and reproducibility. Parameters and numerical outputs are in `results/`.

The assignment heating protocol starts ordered. Random starts are additional fixed-temperature diagnostics. Finite-temperature disorder permits local correlations, and critical or metastable trajectories require careful equilibration checks. The source assignment and lecture notes are not redistributed; check course publication rules before making the project public.
