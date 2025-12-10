#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <iostream>

namespace lsm {

// Basic configuration types

struct ModelParams {
    double S0;      // initial spot
    double r;       // risk-free rate
    double sigma;   // volatility
};

struct OptionParams {
    double K;       // strike
    double T;       // maturity (years)
    bool is_call;   // true = call, false = put
};

struct LsmConfig {
    std::size_t num_paths = 100000;
    std::size_t num_steps = 50;
    unsigned long long rng_seed = 42;
    // Polynomial degree for basis functions: 1 => {1, S}, 2 => {1, S, S^2}, etc.
    std::size_t poly_degree = 2;
};

// ------------------------
// Utility: payoff
// ------------------------

inline double payoff(const OptionParams& opt, double S) {
    if (opt.is_call) {
        return std::max(S - opt.K, 0.0);
    } else {
        return std::max(opt.K - S, 0.0);
    }
}

// POSSIBLE SIMULATION 2
//1: simulate GBM paths at risk-neutral measure

// Returns a matrix paths[path][time_index]
// with time_index = 0to num_steps, where time 0 =S0 and time M=T.
//
inline std::vector<std::vector<double>>
simulate_paths(const ModelParams& model,
               const OptionParams& opt,
               const LsmConfig& cfg)
{
    std::size_t N = cfg.num_paths;
    std::size_t M = cfg.num_steps;
    double dt = opt.T / static_cast<double>(M);

    std::vector<std::vector<double>> paths(N, std::vector<double>(M + 1));
    std::mt19937_64 rng(cfg.rng_seed);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    double drift = (model.r - 0.5 * model.sigma * model.sigma) * dt;
    double vol_sqrt_dt = model.sigma * std::sqrt(dt);

    for (std::size_t i = 0; i < N; ++i) {
        paths[i][0] = model.S0;
        for (std::size_t j = 1; j <= M; ++j) {
            double z = std_normal(rng);
            double log_growth = drift + vol_sqrt_dt * z;
            paths[i][j] = paths[i][j - 1] * std::exp(log_growth);
        }
    }

    return paths;
}


// Utility: small dense linear system solver (Gaussian elimination)
//
// Solves A x = b for x, where A is n x n, b is length n.
// A is given as flat row-major vector of size n*n.
//
inline std::vector<double>
solve_linear_system(std::vector<double> A, std::vector<double> b)
{
    std::size_t n = b.size();
    if (A.size() != n * n) {
        throw std::runtime_error("solve_linear_system: dimension mismatch");
    }

    // Augment A with b: [A | b]
    // We'll perform naive Gaussian elimination with partial pivoting.
    for (std::size_t i = 0; i < n; ++i) {
        // Pivot selection (partial pivoting)
        std::size_t pivot = i;
        double max_abs = std::fabs(A[i * n + i]);
        for (std::size_t r = i + 1; r < n; ++r) {
            double val = std::fabs(A[r * n + i]);
            if (val > max_abs) {
                max_abs = val;
                pivot = r;
            }
        }
        if (max_abs < 1e-14) {
            throw std::runtime_error("solve_linear_system: singular matrix");
        }
        if (pivot != i) {
            // Swap rows in A
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(A[i * n + c], A[pivot * n + c]);
            }
            // Swap in b
            std::swap(b[i], b[pivot]);
        }

        // Eliminate below pivot
        double pivot_val = A[i * n + i];
        for (std::size_t r = i + 1; r < n; ++r) {
            double factor = A[r * n + i] / pivot_val;
            if (std::fabs(factor) < 1e-18) continue;
            for (std::size_t c = i; c < n; ++c) {
                A[r * n + c] -= factor * A[i * n + c];
            }
            b[r] -= factor * b[i];
        }
    }

    // Back substitution
    std::vector<double> x(n, 0.0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = b[i];
        for (std::size_t c = i + 1; c < n; ++c) {
            sum -= A[i * n + c] * x[c];
        }
        double diag = A[i * n + i];
        if (std::fabs(diag) < 1e-14) {
            throw std::runtime_error("solve_linear_system: zero diagonal");
        }
        x[i] = sum / diag;
    }
    return x;
}


// Basis functions: polynomial in S
inline std::vector<double> evaluate_basis(double S, std::size_t degree) {
    std::vector<double> phi(degree + 1);
    double power = 1.0;
    for (std::size_t k = 0; k <= degree; ++k) {
        phi[k] = power;
        power *= S;
    }
    return phi;
}

// POSSIBLE PARALLELISM 1
// Core: Longstaff-Schwartz American option pricer

inline double price_american_lsm(const ModelParams& model,
                                 const OptionParams& opt,
                                 const LsmConfig& cfg)
{
    std::size_t N = cfg.num_paths;
    std::size_t M = cfg.num_steps;
    double dt = opt.T / static_cast<double>(M);
    double disc = std::exp(-model.r * dt);

    // S1. Simulate all paths
    auto paths = simulate_paths(model, opt, cfg);







    // S2. Initialize cashflows at maturity
    std::vector<double> cashflow(N, 0.0);
    std::vector<std::size_t> exercise_index(N, M); // time index of exercise
    // POSSIBLE PARALLELISM 1.1
    for (std::size_t i = 0; i < N; ++i) {
        cashflow[i] = payoff(opt, paths[i][M]);
        // if it's zero, it just means never in the money, that's fine
    }

    // S3. Backward induction over time steps M-1, ..., 1
    const std::size_t deg = cfg.poly_degree;
    const std::size_t K = deg + 1;

    std::vector<std::size_t> itm_paths;  // indices of in-the-money paths
    itm_paths.reserve(N);
    
    for (int j = static_cast<int>(M) - 1; j >= 1; --j) {
        itm_paths.clear();

        // Find paths that are STILL alive and in-the-money at step j
        // POSSIBLE PARALLELISM 1.2
        for (std::size_t i = 0; i < N; ++i) {
            // If exercise_index[i] <= j, it means the option has already been exercised before/at j in backward logic.
            if (exercise_index[i] <= static_cast<std::size_t>(j)) {
                continue;
            }
            double S = paths[i][j];
            double ex_payoff = payoff(opt, S);
            if (ex_payoff > 0.0) {
                itm_paths.push_back(i);
            }
        }

        if (itm_paths.size() < K) {
            // Not enough points to run a stable regression; skip this time step!
            continue;
        }

        // Build normal equations: A = X^T X, b = X^T Y
        std::vector<double> A(K * K, 0.0);
        std::vector<double> b(K, 0.0);


        // POSSIBLE PARALLELISM 1.3
        for (std::size_t idx : itm_paths) {
            double S = paths[idx][j];
            // Discount future cashflow from exercise_index[idx] back to time j
            std::size_t ex_idx = exercise_index[idx];
            double tau = static_cast<double>(ex_idx - j);
            double Y = cashflow[idx] * std::exp(-model.r * dt * tau);

            auto phi = evaluate_basis(S, deg); // length K

            // Accumulate A += phi * phi^T, b += phi * Y
            for (std::size_t r = 0; r < K; ++r) {
                for (std::size_t c = 0; c < K; ++c) {
                    A[r * K + c] += phi[r] * phi[c];
                }
                b[r] += phi[r] * Y;
            }
        }

        // Solve A beta = b
        std::vector<double> beta;
        try {
            beta = solve_linear_system(A, b);
        } catch (const std::exception& e) {
            // If regression fails (singular matrix), skip the early exercise at this time step
            // This is a conservative choice and keeps the price low-biased.
            continue;
        }

        // Now make exercise decisions for all alive in-the-money paths
        // POSSIBLE PARALLELISM 1.3
        for (std::size_t idx : itm_paths) {
            double S = paths[idx][j];
            double ex_payoff = payoff(opt, S);
            if (ex_payoff <= 0.0) continue;

            auto phi = evaluate_basis(S, deg);

            double cont_est = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                cont_est += beta[k] * phi[k];
            }

            if (ex_payoff >= cont_est) {
                // Exercise now: overwrite cashflow and exercise index
                cashflow[idx] = ex_payoff;
                exercise_index[idx] = static_cast<std::size_t>(j);
            }
        }
    }

    // S4. Discount all cashflows to time 0 and average
    // POSSIBLE PARALLELISM 1.4
    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t j = exercise_index[i];
        double t = dt * static_cast<double>(j);
        sum += cashflow[i] * std::exp(-model.r * t);
    }

    return sum / static_cast<double>(N);
}

} // namespace lsm