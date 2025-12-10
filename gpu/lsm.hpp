// lsm.hpp
#pragma once

#include <cstddef>
#include <string>

struct ModelParams {
    double S0;
    double r; 
    double sigma;
};

struct OptionParams {
    double K; 
    double T; 
    bool   is_call; 
};

struct LsmConfig {
    std::size_t num_paths   = 100000;
    std::size_t num_steps   = 50;
    std::size_t poly_degree = 2; 
    unsigned long long rng_seed = 42ULL; 
};

double price_american_lsm_gpu(const ModelParams& model,
                              const OptionParams& opt,
                              const LsmConfig&    cfg,
                              double* total_ms  = nullptr,
                              double* kernel_ms = nullptr);


inline bool parse_option_type(const std::string& s) {
    if (s == "call" || s == "CALL" || s == "c") return true;
    if (s == "put"  || s == "PUT"  || s == "p") return false;
    return true;
}
