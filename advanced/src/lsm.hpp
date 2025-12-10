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
#include "merge_sort.hpp"

#define ROOT 0
#define REBALANCE_INTERVAL 3

#define clz(x) __builtin_clz(x)

int32_t ilog2(uint32_t x){
    return sizeof(uint32_t) * CHAR_BIT - clz(x) - 1;
}

static inline int ceilDiv(int a, int b){
  return (a - 1) / b + 1;
}

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

inline void merge_final_results(
    double sum, 
    double N, 
    int level, 
    int pid, 
    std::vector<double>& buff, 
    int num_levels, 
    std::vector<MPI_Request>& recv_request, 
    std::vector<MPI_Status>& recv_status
){
    int parent = pid & ~(1 << level);
    int child = pid | (1<<(level-1));

    if(level){
        MPI_Wait(&(recv_request[level-1]), &(recv_status[level-1]));

        double child_sum = buff[level * 2];
        double child_N = buff[level * 2 + 1];

        sum += child_sum;
        N += child_N;
    }

    if((level == (num_levels - 1)) || (parent != pid)){
        //put result into buff
        buff[0] = sum;
        buff[1] = N;
    }

    //send to parent
    if(level != (num_levels - 1)){
        if(pid != parent){
            MPI_Send(&buff[0], 2, MPI_DOUBLE, parent, 0, MPI_COMM_WORLD);
        } else{
            merge_final_results(sum, N,level+1, pid, buff, num_levels, recv_request, recv_status);
        }
    }
}

inline void balance_load(
    std::vector<Path_data>& path_datas, 
    std::vector<double>& MS_buff, 
    int num_merges, 
    int num_paths,
    int nproc,
    int pid,
    int N,
    std::vector<double>& recv_from_root_buff,
    std::vector<double>& root_send_buff,
    std::vector<double>& send_paths_buff,
    std::vector<double>& recv_paths_buff,
    std::vector<std::vector<double>>& paths,
    std::vector<std::vector<double>>& owned_paths,
    int path_length,
    int start_time
){

    int max_paths_per_proc = ceilDiv(num_paths, nproc);

    std::vector<MPI_Request> MS_recv_request(num_merges);

    std::vector<MPI_Status> MS_recv_status(num_merges);

    //post recvs for merge_sort
    int MS_msg_size = max_paths_per_proc * 5;
    int MS_offset = max_paths_per_proc * 5;
    for(int i = 1; i <= num_merges; i++){
        int right_child = pid | (1<<(i-1));

        MPI_Irecv(&MS_buff[MS_offset], MS_msg_size, MPI_DOUBLE, right_child, 0, MPI_COMM_WORLD, &MS_recv_request[i-1]);
        
        MS_offset += MS_msg_size;
        MS_msg_size *= 2;
    }

    //post recvs from root
    MPI_Request root_recv_request;
    MPI_Status root_recv_status;
    MPI_Irecv(recv_from_root_buff.data(), N * 5 + 2 * nproc, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD, &root_recv_request);

    //merge sort the path_datas based on value
    
    merge_sort(path_datas, 0, pid, MS_buff, ilog2(nproc)+1, MS_recv_request, MS_recv_status, max_paths_per_proc, num_paths, nproc);
    // printf("finished merge_sort!!!!!!!!!!!!!!, pid: %d\n",pid);

    //wait on sends/recvs from merge_sort
    MPI_Waitall(num_merges, MS_recv_request.data(), MS_recv_status.data());

    //root determines assignments and sends them out
    std::vector<MPI_Request> ROOT_send_request(nproc * (pid == ROOT));

    std::vector<MPI_Status> ROOT_send_status(nproc * (pid == ROOT));
    if(pid == ROOT){

        int msg_size = 5 * max_paths_per_proc + 2 * nproc;

        //pack buffer
        std::vector<std::vector<int>> num_to_send(nproc, std::vector<int>(nproc, 0));
        std::vector<std::vector<int>> num_to_recv(nproc, std::vector<int>(nproc, 0));

        int MS_buff_offset = 0;

        double prev_val = 0;
        for(int i = 0; i < num_paths; i++){
            int worker = i % nproc;

            int global_idx = int(MS_buff[MS_buff_offset + 1]);
            int owner = global_idx % nproc;

            double val = MS_buff[MS_buff_offset];
            if(val < prev_val){
                printf("ERROR not sorted\n");
            }
            prev_val = val;
            
            num_to_recv[worker][owner]++; //worker is going to recv this from owner
            num_to_send[owner][worker]++; //owner is going to send to worker

            int local_idx = global_idx / nproc;

            int offset = owner * msg_size + 2 * nproc + local_idx * 5;

            for(int j = 0; j < 5; j++){
                root_send_buff[offset+j] = (j==4) ? worker : MS_buff[MS_buff_offset];
                MS_buff_offset++;
            }
        }

        for(int i = 0; i < nproc; i++){
            for(int j = 0; j < nproc; j++){
                root_send_buff[i * msg_size + j] = num_to_recv[i][j];
                root_send_buff[i * msg_size + nproc + j] = num_to_send[i][j];
            }
        }

        //send from buffer
        for(int recver = 0; recver < nproc; recver++){
            int recver_num_paths = (num_paths / nproc) + ((num_paths % nproc) > recver);
            MPI_Isend(&root_send_buff[recver * msg_size], (2*nproc + 5 * recver_num_paths), MPI_DOUBLE, recver, 0, MPI_COMM_WORLD, &ROOT_send_request[recver]);
        }
    }

    // // //wait on message from the root
    MPI_Wait(&root_recv_request, &root_recv_status);

    std::vector<int> send_offset(nproc,0);
    std::vector<int> send_offset_begs(nproc, 0);
    std::vector<int> recv_offset(nproc,0);
    std::vector<int> num_to_recv(nproc,0);

    // // ///////////process message from ROOT/////////////////
    // // //process how much to recv from each proc
    int msg_size = 5 + path_length;
    int sum = 0;
    for(int i = 0; i < nproc; i++){
        recv_offset[i] = sum;
        sum += recv_from_root_buff[i] * msg_size;
        num_to_recv[i] = recv_from_root_buff[i];
    }

    std::vector<MPI_Request> recv_paths_request(nproc);

    std::vector<MPI_Status> recv_paths_status(nproc);

    // // //post recvs from every other proc to get my assignment
    for(int i = 0; i < nproc; i++){
        MPI_Irecv(&recv_paths_buff[recv_offset[i]], num_to_recv[i] * msg_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &recv_paths_request[i]);
    }

    // // //process how much to send to each proc
    sum = 0;
    for(int i = 0; i < nproc; i++){
        send_offset[i] = sum;
        send_offset_begs[i] = sum;
        sum += recv_from_root_buff[i + nproc] * msg_size;
    }


    // // // //process path data
    for(int i = 0; i < N; i++){
        int offset = 2 * nproc + i * 5;

        int worker = static_cast<int>(recv_from_root_buff[offset+4]); 
        int global_idx = static_cast<int>(recv_from_root_buff[offset+1]);
        int local_idx = global_idx / nproc;

        for(int j = 0; j < 5; j++){
            send_paths_buff[send_offset[worker]] = recv_from_root_buff[offset+j];
            send_offset[worker]++;
        }

        for(int j = start_time; j < start_time+path_length; j++){
            send_paths_buff[send_offset[worker]] = owned_paths[local_idx][j];
            send_offset[worker]++;
        }
    }

    // // //send out the paths I own to other processors
    std::vector<MPI_Request> send_paths_request(nproc);

    std::vector<MPI_Status> send_paths_status(nproc);

    for(int i = 0; i < nproc; i++){

        MPI_Isend(&send_paths_buff[send_offset_begs[i]], send_offset[i]-send_offset_begs[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &send_paths_request[i]);
    }

    // // //wait on all sends/recvs
    if(pid == ROOT){

        MPI_Waitall(nproc, ROOT_send_request.data(), ROOT_send_status.data());
    }

    MPI_Waitall(nproc, send_paths_request.data(), send_paths_status.data());

    MPI_Waitall(nproc, recv_paths_request.data(), recv_paths_status.data());

    // // //build vector of data and paths for the paths that are assigned to me
    int offset = 0;
    for(int i = 0; i < 2*N; i++){
        //read the path_data
        if(!(i % 2)){
            double val = recv_paths_buff[offset];
            double global_idx = recv_paths_buff[offset+1];
            double cashflow = recv_paths_buff[offset+2];
            double exercise_idx = recv_paths_buff[offset+3];
            double worker = recv_paths_buff[offset+4];

            path_datas[i/2] = Path_data{val, global_idx, cashflow, exercise_idx, worker};

            offset+=5;
        } else{ //read the path
            for(int j = 0; j < path_length; j++){
                paths[i/2][j] = recv_paths_buff[offset];
                offset++;
            }
        }
    }
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

    // S1. Simulate all paths

    //how many paths this processor simulates
    int num_paths_assigned = (static_cast<int>(N) / nproc) + ((static_cast<int>(N) % nproc) > pid);
    N = static_cast<std::size_t>(num_paths_assigned);

    //simulate paths with global index pid, pid+nproc, pid+2nproc, ...

    auto owned_paths = simulate_paths(model, opt, cfg, N);



    // S2. Initialize cashflows at maturity
    // std::vector<double> cashflow(N, 0.0);
    // std::vector<std::size_t> exercise_index(N, M); // time index of exercise

    std::vector<Path_data> path_datas(N);
    for (std::size_t i = 0; i < N; ++i) {

        double val = owned_paths[i][M];
        double global_idx = pid+(nproc*i);

        double cashflow = payoff(opt, owned_paths[i][M]);
        double exercise_idx = M; 
        path_datas[i] = Path_data{val, global_idx, cashflow, exercise_idx, static_cast<double>(pid)};
    }

    ////////////// allocate buffers for load balancing /////////////////

    // int max_path_length = M-1;
    int max_path_length = REBALANCE_INTERVAL;

    int max_paths_per_proc = ceilDiv(cfg.num_paths, nproc);

    //buff for send/recv in merge_sort
    //message size doubles on each level if you're doing the merging
    int msg_size = 5 * max_paths_per_proc;
    int MS_buff_size = 5 * max_paths_per_proc;
    int num_merges = 0;
    for(int i = 1; i < ilog2(nproc)+1; i++){
        if(!(pid % (1 << i))){
            MS_buff_size += msg_size;
            msg_size *= 2;
            num_merges++;
        }
    }

    std::vector<double> MS_buff(MS_buff_size);
    
    std::vector<double> recv_from_root_buff(N * 5 + 2*nproc);

    std::vector<double> root_send_buff((max_paths_per_proc * 5 + 2*nproc) * nproc * (pid == ROOT));

    std::vector<double> send_paths_buff(N * (max_path_length + 5));

    std::vector<double> recv_paths_buff(N * (max_path_length + 5));
    //////////////////////////////////////////////////////////////////

    std::vector<std::vector<double>> paths(N, std::vector<double>(max_path_length,0));


    // S3. Backward induction over time steps M-1, ..., 1
    const std::size_t deg = cfg.poly_degree;
    const std::size_t K = deg + 1;

    std::vector<std::size_t> itm_paths;  // indices of in-the-money paths
    itm_paths.reserve(N);

    int start_time;
    int path_length;
    
    for (int k = static_cast<int>(M) - 1; k >= 1; --k) {
        if((k == static_cast<int>(M)-1) || ((k % REBALANCE_INTERVAL)==0)){
            
            start_time = std::max(0, k - (REBALANCE_INTERVAL - 1));
            path_length = k - start_time + 1;

            // if (pid == 0) {
            //     printf("k=%d  start=%d  length=%d\n", k, start_time, path_length);
            //     for (int i = 0; i < 5; i++) {
            //         printf("Before balance: S[%d][k]=%f\n", i, owned_paths[i][k]);
            //     }
            // }

            balance_load(path_datas, MS_buff, num_merges, cfg.num_paths,
                    nproc, pid, N, recv_from_root_buff, root_send_buff,
                    send_paths_buff, recv_paths_buff, paths, owned_paths,
                    path_length, start_time);

            // if (pid == 0) {
            //     for (int i = 0; i < 5; i++) {
            //         printf("After balance:  paths[%d][j_last]=%f\n", i, paths[i][path_length-1]);
            //     }
            // }
        }
        int j = k - start_time;

        itm_paths.clear();

        // Find paths that are STILL alive and in-the-money at step j
        // POSSIBLE PARALLELISM 1.2
        for (std::size_t i = 0; i < N; ++i) {
            // If exercise_index[i] <= j, it means the option has already been exercised before/at j in backward logic.
            if (path_datas[i].exercise_idx <= static_cast<std::size_t>(k)) {
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
            std::size_t ex_idx = path_datas[idx].exercise_idx;
            double tau = static_cast<double>(ex_idx - k);
            double Y = path_datas[idx].cashflow * std::exp(-model.r * dt * tau);

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
            for (std::size_t l = 0; l < K; ++l) {
                cont_est += beta[l] * phi[l];
            }

            if (ex_payoff >= cont_est) {
                // Exercise now: overwrite cashflow and exercise index
                path_datas[idx].cashflow = ex_payoff;
                path_datas[idx].exercise_idx = static_cast<std::size_t>(k);
            }
        }
    }

    // S4. Discount all cashflows to time 0 and average
    // POSSIBLE PARALLELISM 1.4
    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t j = path_datas[i].exercise_idx;
        double t = dt * static_cast<double>(j);
        sum += path_datas[i].cashflow * std::exp(-model.r * t);
    }

    //merge final results together
    std::vector<double> final_results_buff(2 + 2 * num_merges);

    std::vector<MPI_Request> final_results_recv_request(num_merges);

    std::vector<MPI_Status> final_results_recv_status(num_merges);

    //post recvs for merge_final_results
    int final_results_offset = 2;
    for(int i = 1; i <= num_merges; i++){
        int right_child = pid | (1<<(i-1));

        MPI_Irecv(&final_results_buff[final_results_offset], 2, MPI_DOUBLE, right_child, 0, MPI_COMM_WORLD, &final_results_recv_request[i-1]);
        
        final_results_offset += 2;
    }

    merge_final_results(sum, N, 0, pid, final_results_buff, ilog2(nproc)+1, final_results_recv_request, final_results_recv_status);

    MPI_Waitall(num_merges, final_results_recv_request.data(), final_results_recv_status.data());

    if(pid == ROOT){
        //recv all other averages and return total average
        // return sum / static_cast<double>(N);
        return final_results_buff[0] / final_results_buff[1];
    } else{
        return 0;
        // return sum / static_cast<double>(N);
    }
}

} // namespace lsm