// main.cpp
#include "lsm.hpp"

#include <getopt.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

void usage(const char* prog) {
    std::printf("Usage: %s [options]\n", prog);
    std::printf("Options:\n");
    std::printf("  --S0 <double>      Initial spot (default 100)\n");
    std::printf("  --K  <double>      Strike (default 100)\n");
    std::printf("  --r  <double>      Risk-free rate (default 0.05)\n");
    std::printf("  --sigma <double>   Volatility (default 0.2)\n");
    std::printf("  --T  <double>      Maturity in years (default 1.0)\n");
    std::printf("  --paths <int>      Number of Monte Carlo paths (default 100000)\n");
    std::printf("  --steps <int>      Number of exercise steps (default 50)\n");
    std::printf("  --deg <int>        Polynomial degree (default 2)\n");
    std::printf("  --seed <int>       RNG seed (default 42)\n");
    std::printf("  --call             Price American call (default)\n");
    std::printf("  --put              Price American put\n");
    std::printf("  -?  --help         This message\n");
}

int main(int argc, char** argv)
{
    ModelParams model{100.0, 0.05, 0.2};
    OptionParams opt{100.0, 1.0, true};   // default = CALL
    LsmConfig cfg;

    // parse options
    int optch;
    static struct option long_opts[] = {
        {"S0",    1, 0, 0},
        {"K",     1, 0, 0},
        {"r",     1, 0, 0},
        {"sigma", 1, 0, 0},
        {"T",     1, 0, 0},
        {"paths", 1, 0, 0},
        {"steps", 1, 0, 0},
        {"deg",   1, 0, 0},
        {"seed",  1, 0, 0},
        {"call",  0, 0, 0},
        {"put",   0, 0, 0},
        {"help",  0, 0, '?'},
        {0,0,0,0}
    };
    int long_index = 0;

    while ((optch = getopt_long(argc, argv, "?", long_opts, &long_index)) != -1) {
        if (optch == '?') {
            usage(argv[0]);
            return 0;
        }
        if (optch == 0) {
            std::string name(long_opts[long_index].name);
            if (name == "S0")         model.S0   = std::atof(optarg);
            else if (name == "K")     opt.K      = std::atof(optarg);
            else if (name == "r")     model.r    = std::atof(optarg);
            else if (name == "sigma")model.sigma = std::atof(optarg);
            else if (name == "T")     opt.T      = std::atof(optarg);
            else if (name == "paths")cfg.num_paths = std::strtoull(optarg, nullptr, 10);
            else if (name == "steps")cfg.num_steps = std::strtoull(optarg, nullptr, 10);
            else if (name == "deg")  cfg.poly_degree = std::strtoull(optarg, nullptr, 10);
            else if (name == "seed") cfg.rng_seed    = std::strtoull(optarg, nullptr, 10);
            else if (name == "call") opt.is_call = true;
            else if (name == "put")  opt.is_call = false;
        }
    }

    std::cout << "GPU Longstaff–Schwartz Monte Carlo\n";
    std::cout << "  S0    = " << model.S0 << "\n"
              << "  K     = " << opt.K << "\n"
              << "  r     = " << model.r << "\n"
              << "  sigma = " << model.sigma << "\n"
              << "  T     = " << opt.T << "\n"
              << "  paths = " << cfg.num_paths << "\n"
              << "  steps = " << cfg.num_steps << "\n"
              << "  deg   = " << cfg.poly_degree << "\n"
              << "  seed  = " << cfg.rng_seed << "\n"
              << "  type  = " << (opt.is_call ? "call" : "put") << "\n"
              << std::endl;

    double total_ms = 0.0, kernel_ms = 0.0;
    double price = price_american_lsm_gpu_perblock(model, opt, cfg, &total_ms, &kernel_ms);

    std::cout << "Estimated American option price: " << price << "\n";
    std::cout << "Total GPU time  (ms): " << total_ms  << "\n";
    std::cout << "Path sim GPU time (ms): " << kernel_ms << "\n";

    return 0;
}
