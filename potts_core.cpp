#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// Single-spin Metropolis updates, with periodic boundaries and J = k_B = 1.
// The zero-field energy is -3/2 per equal nearest-neighbour bond.
extern "C" void potts_run(
    const double* temperatures, int n_t, int size, double field, int initial_state,
    int burn_sweeps, int samples, int sample_interval, std::uint64_t seed,
    double* magnetisation, double* mx, double* energy, double* acceptance,
    std::int8_t* snapshots
) {
    const int n = size * size;
    const double sx[3] = {1.0, -0.5, -0.5};
    const double sy[3] = {0.0, std::sqrt(3.0) / 2.0, -std::sqrt(3.0) / 2.0};
    std::vector<int> spins(n, std::max(initial_state, 0));
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> site(0, n - 1), alternative(1, 2);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const int ordered_state = std::max(initial_state, 0);
    double sum_x = n * sx[ordered_state], sum_y = n * sy[ordered_state];
    double total_energy = -3.0 * n - field * sum_x;
    if (initial_state == -1) {
        // Random starts need their actual observables, not ordered-state values.
        std::uniform_int_distribution<int> state(0, 2);
        sum_x = sum_y = 0.0;
        for (int& spin : spins) {
            spin = state(rng);
            sum_x += sx[spin];
            sum_y += sy[spin];
        }
        int matches = 0;
        for (int i = 0; i < n; ++i) {
            int row = i / size, col = i % size;
            matches += spins[i] == spins[row * size + (col + 1) % size];
            matches += spins[i] == spins[((row + 1) % size) * size + col];
        }
        total_energy = -1.5 * matches - field * sum_x;
    }

    for (int t = 0; t < n_t; ++t) {
        std::uint64_t accepted = 0, attempted = 0;
        auto sweep = [&](bool measure_acceptance) {
            for (int attempt = 0; attempt < n; ++attempt) {
                int i = site(rng), row = i / size, col = i % size;
                int old = spins[i], proposed = (old + alternative(rng)) % 3;
                int neighbours[4] = {
                    ((row + size - 1) % size) * size + col,
                    ((row + 1) % size) * size + col,
                    row * size + (col + size - 1) % size,
                    row * size + (col + 1) % size
                };
                int old_matches = 0, new_matches = 0;
                for (int j : neighbours) {
                    old_matches += spins[j] == old;
                    new_matches += spins[j] == proposed;
                }
                double dx = sx[proposed] - sx[old];
                double dy = sy[proposed] - sy[old];
                double delta = -1.5 * (new_matches - old_matches) - field * dx;
                bool accept = delta <= 0.0 || uniform(rng) < std::exp(-delta / temperatures[t]);
                if (accept) {
                    spins[i] = proposed;
                    sum_x += dx;
                    sum_y += dy;
                    total_energy += delta;
                }
                if (measure_acceptance) {
                    ++attempted;
                    accepted += accept;
                }
            }
        };
        for (int k = 0; k < burn_sweeps; ++k) sweep(false);
        for (int k = 0; k < samples; ++k) {
            for (int step = 0; step < sample_interval; ++step) sweep(true);
            int index = t * samples + k;
            magnetisation[index] = std::hypot(sum_x, sum_y) / n;
            mx[index] = sum_x / n;
            energy[index] = total_energy / n;
        }
        acceptance[t] = static_cast<double>(accepted) / attempted;
        for (int i = 0; i < n; ++i) snapshots[t * n + i] = spins[i];
    }
}
