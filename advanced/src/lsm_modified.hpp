#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <utility>
#include <mpi.h>
#include <ranges>
#include <climits>
#include <numeric>
#include "merge_sort.hpp"
#include <sstream>


#define ROOT 0
#define PATHS_PER_ITERATION 100

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
               const LsmConfig& cfg,
               const std::size_t N)
{
    // std::size_t N = cfg.num_paths;
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
                                 const LsmConfig& cfg,
                                 const int pid,
                                 const int nproc)
{
    std::size_t N = cfg.num_paths;
    std::size_t M = cfg.num_steps;
    double dt = opt.T / static_cast<double>(M);
    double disc = std::exp(-model.r * dt);

    double lambda = 2.0;
    double mu = 2.0;
    double nu = 0.99;

    //1. Initiate the algorithm using rough estimates for exercise boundaries or coeﬃcients
    // consider that the option is exercised at final maturity only

    //set to all 0's
    std::vector<std::vector<double>> prev_alpha(M, std::vector<double>(cfg.poly_degree, 1e-10));
    std::vector<double> prev_alpha_flat((M) * cfg.poly_degree, 1e-10);

    //vector of num_time_steps degxdeg matrices flattened
    int num_time_steps = std::floor(((static_cast<int>(M)) * (pid + 1) / nproc)) - std::floor((static_cast<int>(M)) * pid / nproc);
    std::vector<std::vector<double>> U(num_time_steps,std::vector<double>(cfg.poly_degree * cfg.poly_degree, 0));

    //vector of num_time_steps degx1 vectors
    std::vector<std::vector<double>> V(num_time_steps,std::vector<double>(cfg.poly_degree, 0));

    std::vector<std::vector<double>> local_alpha(num_time_steps, std::vector<double>(cfg.poly_degree,0));

    double P_bar = 0;
    double q = 0;


    //2. Iterate on i from 1 to n:
    int n = static_cast<int>(cfg.num_paths) / PATHS_PER_ITERATION;

    for(int i = 1; i <= n; i++){

        int start_idx = (i-1) * PATHS_PER_ITERATION;
        int global_num_paths = std::min(PATHS_PER_ITERATION, static_cast<int>(N) - start_idx);
        int local_num_paths = std::floor(global_num_paths * (pid+1)/nproc) - std::floor(global_num_paths * pid / nproc);

        auto paths = simulate_paths(model, opt, cfg, local_num_paths);

        double w_tilde = 1 - (.5 * (1 - tanh(nu * (static_cast<double>(i)-1))));

        double w = 1.0;
        for(int j = i+1; j <= n; j++){
            w *= 1 - (lambda * std::exp(-static_cast<double>(j)/mu));
        }
        

        double price = 0;

        //vector of M degx1 matrices
        std::vector<std::vector<std::vector<double>>> u(M,std::vector<std::vector<double>>(cfg.poly_degree, std::vector<double>(cfg.poly_degree, 0)));

        //vector of M degx1 vectors
        std::vector<std::vector<double>> v(M,std::vector<double>(cfg.poly_degree, 0));

        //a

        for(int j = 0; j < local_num_paths; j++){

            std::vector<double> P(M);
            std::vector<double> P_tilde(M);

            // i
            std::vector<double> curr_path = paths[j];

            // ii

            P[M-1] = payoff(opt,curr_path[M-1]);

            for(int k = M-2; k >= 0; k--){
                P_tilde[k+1] = disc * P[k+1];

                double payoff_value = payoff(opt,curr_path[k]);
                double continuation_value = std::inner_product(prev_alpha[k].begin(), prev_alpha[k].end(), evaluate_basis(curr_path[k],cfg.poly_degree).begin(),0.0);

                if(payoff_value > continuation_value){
                    P[k] = payoff_value;
                } else{
                    P[k] = P_tilde[k+1];
                }
            }
            
            // iii
            price += P[0];

            //iv
            for(int k = 0; k < static_cast<int>(M); k++){
                double val = curr_path[k];

                std::vector<double> basis = evaluate_basis(val,cfg.poly_degree);

                for(int a = 0; a < static_cast<int>(cfg.poly_degree); a++){
                    for(int b = 0; b < static_cast<int>(cfg.poly_degree); b++){
                        u[k][a][b] += w * basis[a] * basis[b];
                    }
                    v[k][a] += w * basis[a] * P_tilde[k+1];
                }
            }
        }

        /////////// reduce to get U_k and V_k ///////////////
        std::vector<std::vector<double>> send_buff(nproc);

        std::vector<std::vector<double>> recv_buff(nproc);

        for(int proc = 0; proc < nproc; proc++){
            int n = std::floor(((static_cast<int>(M)) * (proc + 1) / nproc)) - std::floor((static_cast<int>(M)) * proc / nproc);
            send_buff[proc] = std::vector<double>((static_cast<int>(cfg.poly_degree) * static_cast<int>(cfg.poly_degree) + static_cast<int>(cfg.poly_degree)) * n,0);
            recv_buff[proc] = std::vector<double>((static_cast<int>(cfg.poly_degree) * static_cast<int>(cfg.poly_degree) + static_cast<int>(cfg.poly_degree)) * n,0);
        }

        std::vector<MPI_Request> request(nproc);
        std::vector<MPI_Status> status(nproc);

        // debug_print_u_v_before("BEFORE REDUCTION", u, v, MPI_COMM_WORLD);

        for(int proc = 0; proc < nproc; proc++){
            //pack the buffer
            int start_time = std::floor((static_cast<int>(M)-1) * proc / nproc);
            int end_time = start_time + std::floor(((static_cast<int>(M)) * (proc + 1) / nproc)) - std::floor((static_cast<int>(M)) * proc / nproc);

            int offset = 0;

            for(int t = start_time; t < end_time; t++){
                //pack u
                for(int a = 0; a < static_cast<int>(cfg.poly_degree); a++){
                    for(int b = 0; b < static_cast<int>(cfg.poly_degree); b++){
                        send_buff[proc][offset] = u[t][a][b];
                        offset ++;
                    }
                }

                //pack v
                for(int a = 0; a < static_cast<int>(cfg.poly_degree); a++){
                    send_buff[proc][offset] = v[t][a];
                    offset++;
                }
            }

            MPI_Ireduce(
                send_buff[proc].data(),
                recv_buff[proc].data(),
                (static_cast<int>(cfg.poly_degree) * static_cast<int>(cfg.poly_degree) + static_cast<int>(cfg.poly_degree)) * (end_time - start_time),
                MPI_DOUBLE,
                MPI_SUM,
                proc,
                MPI_COMM_WORLD,
                &request[proc]
            );
        }

        //////////////////////////////////////////////////////

        //b

        //wait for our reduction to complete
        MPI_Wait(&(request[pid]), &(status[pid]));


        int start_time = std::floor((static_cast<int>(M)) * pid / nproc);
        int end_time = start_time + std::floor(((static_cast<int>(M)) * (pid + 1) / nproc)) - std::floor((static_cast<int>(M)) * pid / nproc);

        int offset = 0;
        // for(int k = start_time; k < end_time; k++){
        for(int k = 0; k < num_time_steps; k++){
            for(int a = 0; a < static_cast<int>(cfg.poly_degree) * static_cast<int>(cfg.poly_degree); a++){
                U[k][a] += recv_buff[pid][offset];;
                offset++;
            }

            for(int a = 0; a < static_cast<int>(cfg.poly_degree); a++){
                V[k][a] += recv_buff[pid][offset];
                offset++;
            }
            // printf("about to solve from pid: %d, i: %d\n",pid,i);
            // local_alpha[k] = solve_linear_system(U[k], V[k]);
            // printf("finished solving from pid: %d\n",pid);

        }
        // debug_print_U_V_after("AFTER REDUCTION", U, V, MPI_COMM_WORLD, cfg.poly_degree);

        //c
        P_bar += w_tilde * price;
        q += w_tilde * global_num_paths;

        //send out and get alphas
        std::vector<int> recv_counts(nproc);
        for(int a = 0; a < nproc; a++){
            recv_counts[a] = (std::floor(((static_cast<int>(M)) * (a + 1) / nproc)) - std::floor((static_cast<int>(M)) * a / nproc)) * static_cast<int>(cfg.poly_degree);
        }

        std::vector<int> displ(nproc);
        int sum = 0;
        for(int a = 0; a < nproc; a++){
            displ[a] = sum;
            sum += recv_counts[a];
        }
        MPI_Request alpha_request;
        MPI_Status alpha_status;

        std::vector<double> local_alpha_flat(static_cast<int>(cfg.poly_degree) * num_time_steps, 0);
        offset = 0;
        for(int t = 0; t < num_time_steps; t++){
            for(int a = 0; a < static_cast<int>(cfg.poly_degree); a++){
                local_alpha_flat[offset] = local_alpha[t][a];
                offset ++;
            }
        }

        MPI_Iallgatherv(
            local_alpha_flat.data(),
            num_time_steps * static_cast<int>(cfg.poly_degree),
            MPI_DOUBLE,
            prev_alpha_flat.data(),
            recv_counts.data(),
            displ.data(),
            MPI_DOUBLE,
            MPI_COMM_WORLD,
            &alpha_request
        );

        //wait for all communication to complete
        MPI_Waitall(nproc, request.data(), status.data());
        MPI_Wait(&alpha_request, &alpha_status);

        //fill in prev_alpha using prev_alpha_flat
        offset = 0;
        for(int t = 0; t < static_cast<int>(M); t++){
            for(int a = 0; a < static_cast<int>(cfg.poly_degree); a++){
                prev_alpha[t][a] = prev_alpha_flat[offset];
                offset++;
            }
        }

    }

    //3. Finally get the Monte Carlo estimate of the option price as the weighted average
    double result;
    MPI_Reduce(&P_bar, &result, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if(pid == ROOT){
        return result / q;
    } else{
        return 0;
    }
}

} // namespace lsm