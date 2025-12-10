// lsm.hpp
#pragma once

#include <cstddef>
#include <string>

// Simple parameter structs, same idea as in your CPU baseline
struct ModelParams {
    double S0;      // initial spot
    double r;       // risk-free rate
    double sigma;   // volatility
};

struct OptionParams {
    double K;       // strike
    double T;       // maturity (years)
    bool   is_call; // true = call, false = put
};

struct LsmConfig {
    std::size_t num_paths   = 100000;
    std::size_t num_steps   = 50;
    std::size_t poly_degree = 2;         // polynomial degree for basis
    unsigned long long rng_seed = 42ULL; // RNG seed
};

// GPU driver: run full Longstaff–Schwartz on GPU and return price.
// Optionally returns kernel-only and total time in milliseconds.
double price_american_lsm_gpu_perblock(const ModelParams& model,
                                       const OptionParams& opt,
                                       const LsmConfig& cfg,
                                       double* total_ms,
                                       double* kernel_ms);
// Small helper to convert option type string to bool
inline bool parse_option_type(const std::string& s) {
    if (s == "call" || s == "CALL" || s == "c") return true;
    if (s == "put"  || s == "PUT"  || s == "p") return false;
    // default: call
    return true;
}
