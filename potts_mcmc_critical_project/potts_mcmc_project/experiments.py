"""Reproduce the heating experiments and figures for the three-state Potts model."""
from pathlib import Path
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from fast_potts import simulate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

TEMPERATURES = np.linspace(0.6, 2.4, 25)
TC = 1.5 / np.log(1 + np.sqrt(3))
SEED = 20261027


def experiment(size, field, initial_state, repetitions=8, burn=500, samples=500, interval=1, seed_offset=0):
    """One observation is a complete, independently seeded heating run."""
    mean_m, mean_mx, mean_e, accepts, pictures = [], [], [], [], []
    rng = np.random.default_rng(SEED + seed_offset)
    for rep in range(repetitions):
        seed = int(rng.integers(0, 2**31 - 1))
        data = simulate(TEMPERATURES, size=size, field=field, initial_state=initial_state,
                        burn_sweeps=burn, samples=samples, sample_interval=interval, seed=seed)
        mean_m.append(data["magnetisation"].mean(axis=1))
        mean_mx.append(data["mx"].mean(axis=1))
        mean_e.append(data["energy"].mean(axis=1))
        accepts.append(data["acceptance"])
        pictures.append(data["snapshots"])
        print(f"L={size}, h={field:g}, initial={initial_state + 1}, run {rep + 1}/{repetitions}", flush=True)
    return {"m": np.array(mean_m), "mx": np.array(mean_mx), "energy": np.array(mean_e),
            "acceptance": np.array(accepts), "snapshots": np.array(pictures)}


def mean_sem(values):
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def errorbar(ax, values, label, color=None, shift=0):
    mean, sem = mean_sem(values)
    ax.errorbar(TEMPERATURES + shift, mean, yerr=sem, label=label, color=color,
                lw=1.3, marker="o", ms=2.7, capsize=2, elinewidth=.8)


def main():
    start = time.time()
    records = {}
    # Each combination has independent entire heating trajectories.
    for fidx, field in enumerate((0.0, 0.08, 0.20)):
        for state in range(3):
            key = f"L24_h{fidx}_s{state}"
            records[key] = experiment(24, field, state, seed_offset=1000 * fidx + 100 * state)
    # Different L probe rounding and finite-size effects; the L=24 reference is reused.
    for size in (12, 36):
        key = f"L{size}_h0_s0"
        records[key] = experiment(size, 0.0, 0, seed_offset=10000 + size)
    # A longer burn-in controls for potential heating-history and equilibration effects.
    records["L24_h2_s1_long"] = experiment(24, 0.20, 1, repetitions=8, burn=2000,
                                           samples=500, seed_offset=20000)
    np.savez_compressed(RESULTS / "simulations.npz", temperatures=TEMPERATURES,
                        **{f"{key}_{k}": v for key, record in records.items() for k, v in record.items()})
    metadata = {"temperature_min": float(TEMPERATURES[0]), "temperature_max": float(TEMPERATURES[-1]),
                "temperature_step": float(TEMPERATURES[1]-TEMPERATURES[0]), "seed": SEED,
                "repetitions": 8, "burn_sweeps": 500, "samples": 500, "sample_interval_sweeps": 1,
                "long_burn_sweeps": 2000, "J": 1, "k_B": 1, "periodic_boundaries": True,
                "field_values": [0, .08, .2], "critical_temperature_exact": TC,
                "estimated_elapsed_seconds": round(time.time()-start, 1)}
    (RESULTS / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_figures(records)
    write_summary(records)
    switching_h1()


def switching_h1():
    """Compare the original four-sweep heating schedule with a longer schedule.

    Record one configuration per temperature, as in the original example.
    Each result is an independently seeded complete heating trajectory.
    """
    temperatures = np.linspace(0.5, 2.5, 20)
    seed_rng = np.random.default_rng(SEED + 30000)
    arrays = {}
    for name, burn in (("short", 3), ("long", 499)):
        seeds = seed_rng.integers(0, 2**31 - 1, size=8)
        magnitudes, components = [], []
        for seed in seeds:
            data = simulate(temperatures, size=50, field=1.0, initial_state=1,
                            burn_sweeps=burn, samples=1, sample_interval=1, seed=int(seed))
            magnitudes.append(data["magnetisation"][:, 0])
            components.append(data["mx"][:, 0])
        arrays[f"{name}_m"] = np.array(magnitudes)
        arrays[f"{name}_mx"] = np.array(components)
        arrays[f"{name}_seeds"] = seeds
    metadata = {"size": 50, "field": 1.0, "initial_state_zero_based": 1,
                "repetitions": 8, "short_burn_sweeps": 3, "long_burn_sweeps": 499,
                "measurements_per_temperature": 1, "measurement_sweeps": 1,
                "J": 1.0, "k_B": 1.0, "periodic_boundaries": True}
    np.savez_compressed(RESULTS / "switching_h1.npz", temperatures=temperatures,
                        metadata=json.dumps(metadata), **arrays)


def make_figures(records):
    colors = ["#27639f", "#e28b32", "#7d569e"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 140, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, constrained_layout=True)
    for fidx, (ax, h) in enumerate(zip(axes, [0, .08, .2])):
        for state, color in enumerate(colors):
            errorbar(ax, records[f"L24_h{fidx}_s{state}"]["m"], f"Initial state {state + 1}", color,
                     shift=(state-1)*0.005)
        if h == 0:
            ax.axvline(TC, color="0.35", lw=1, ls="--", label=r"Exact $T_c$")
        ax.set(title=f"$h={h:g}$", xlabel=r"Temperature $T/J$")
        ax.set_ylim(-.02, 1.05)
        ax.grid(alpha=.16)
    axes[0].set_ylabel(r"Mean magnetisation magnitude $\langle |m| \rangle$")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Heating from three ordered states ($L=24$); error bars: independent-run SEM")
    fig.savefig(FIGURES / "magnetisation_by_field.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for size, color in zip([12, 24, 36], colors):
        errorbar(ax, records[f"L{size}_h0_s0"]["m"], f"$L={size}$", color)
    ax.axvline(TC, color="0.35", lw=1, ls="--", label=r"Exact $T_c$")
    ax.set(xlabel=r"Temperature $T/J$", ylabel=r"$\langle |m| \rangle$", title="Finite-size comparison at zero field")
    ax.grid(alpha=.16)
    ax.legend()
    fig.savefig(FIGURES / "finite_size.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3), constrained_layout=True)
    snapshots = records["L24_h0_s0"]["snapshots"][0]
    for ax, target in zip(axes, [0.75, 1.5, 2.25]):
        i = int(np.argmin(abs(TEMPERATURES - target)))
        ax.imshow(snapshots[i], cmap=ListedColormap(colors), vmin=0, vmax=2, interpolation="nearest")
        ax.set(title=f"$T={TEMPERATURES[i]:.2f}J$"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Example configurations at zero field ($L=24$)")
    fig.savefig(FIGURES / "configurations.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for state, color in enumerate(colors):
        errorbar(axes[0], records[f"L24_h2_s{state}"]["mx"], f"Initial state {state+1}", color)
    axes[0].axhline(0, color="0.5", lw=.7)
    axes[0].set(title="Direction selected by $h=0.20$", xlabel=r"$T/J$", ylabel=r"$\langle m_x \rangle$")
    short = records["L24_h2_s1"]["mx"]
    long = records["L24_h2_s1_long"]["mx"]
    errorbar(axes[1], short, "500 equilibration sweeps", colors[0])
    errorbar(axes[1], long, "2,000 equilibration sweeps", colors[1])
    axes[1].set(title="Equilibration sensitivity ($h=0.20$, state 2)", xlabel=r"$T/J$", ylabel=r"$\langle m_x \rangle$")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.16)
    fig.savefig(FIGURES / "equilibration_and_field.png")
    plt.close(fig)


def write_summary(records):
    rows = []
    for key, data in records.items():
        mean, sem = mean_sem(data["m"])
        mx, _ = mean_sem(data["mx"])
        for i, temp in enumerate(TEMPERATURES):
            rows.append(f"{key},{temp:.3f},{mean[i]:.6f},{sem[i]:.6f},{mx[i]:.6f}")
    (RESULTS / "summary.csv").write_text("experiment,temperature,mean_magnitude,sem,mean_mx\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
