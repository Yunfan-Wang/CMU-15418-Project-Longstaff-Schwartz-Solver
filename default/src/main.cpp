#include "lsm.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <algorithm>   // IMPORTANT: needed for std::find

// Find a flag in argv and extract the following argument as double
double get_arg_double(char** begin, char** end, const std::string& flag, double default_val) {
    for (char** itr = begin; itr != end; ++itr) {
        if (flag == std::string(*itr)) {     // convert to std::string
            if (++itr != end) return std::atof(*itr);
        }
    }
    return default_val;
}

// Find a flag in argv and extract the following argument as size_t
std::size_t get_arg_size_t(char** begin, char** end, const std::string& flag, std::size_t default_val) {
    for (char** itr = begin; itr != end; ++itr) {
        if (flag == std::string(*itr)) {     // convert to std::string
            if (++itr != end) return static_cast<std::size_t>(std::strtoull(*itr, nullptr, 10));
        }
    }
    return default_val;
}

// Check if a flag exists (no value needed)
bool has_flag(char** begin, char** end, const std::string& flag) {
    for (char** itr = begin; itr != end; ++itr) {
        if (flag == std::string(*itr)) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    using namespace lsm;

    if (has_flag(argv, argv + argc, "--help") || has_flag(argv, argv + argc, "-h")) {
        std::cout << "Usage: ./lsm_price [options]\n\n"
                  << "Options:\n"
                  << "  --S0    <double>   Initial spot price (default 100)\n"
                  << "  --K     <double>   Strike price (default 100)\n"
                  << "  --r     <double>   Risk-free rate (default 0.05)\n"
                  << "  --sigma <double>   Volatility (default 0.2)\n"
                  << "  --T     <double>   Maturity in years (default 1.0)\n"
                  << "  --paths <int>      Number of Monte Carlo paths (default 100000)\n"
                  << "  --steps <int>      Number of time steps (default 50)\n"
                  << "  --deg   <int>      Polynomial degree for basis (default 2)\n"
                  << "  --seed  <int>      RNG seed (default 42)\n"
                  << "  --call             Price American call (default)\n"
                  << "  --put              Price American put\n"
                  << std::endl;
        return 0;
    }

    ModelParams model;
    OptionParams opt;
    LsmConfig cfg;

    model.S0 = get_arg_double(argv, argv + argc, "--S0", 100.0);
    opt.K    = get_arg_double(argv, argv + argc, "--K", 100.0);
    model.r  = get_arg_double(argv, argv + argc, "--r", 0.05);
    model.sigma = get_arg_double(argv, argv + argc, "--sigma", 0.2);
    opt.T       = get_arg_double(argv, argv + argc, "--T", 1.0);

    cfg.num_paths = get_arg_size_t(argv, argv + argc, "--paths", 100000);
    cfg.num_steps = get_arg_size_t(argv, argv + argc, "--steps", 50);
    cfg.poly_degree = get_arg_size_t(argv, argv + argc, "--deg", 2);
    cfg.rng_seed = static_cast<unsigned long long>(
        get_arg_size_t(argv, argv + argc, "--seed", 42)
    );

    bool is_call = has_flag(argv, argv + argc, "--put") ? false : true;
    opt.is_call = is_call;

    std::cout << "Running Longstaff-Schwartz baseline (sequential)\n";
    std::cout << "  S0    = " << model.S0 << "\n"
              << "  K     = " << opt.K << "\n"
              << "  r     = " << model.r << "\n"
              << "  sigma = " << model.sigma << "\n"
              << "  T     = " << opt.T << "\n"
              << "  paths = " << cfg.num_paths << "\n"
              << "  steps = " << cfg.num_steps << "\n"
              << "  deg   = " << cfg.poly_degree << "\n"
              << "  seed  = " << cfg.rng_seed << "\n"
              << "  type  = " << (opt.is_call ? "American Call" : "American Put") << "\n"
              << std::endl;

    try {
        double price = price_american_lsm(model, opt, cfg);
        std::cout << "Estimated American option price: " << price << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during pricing: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}