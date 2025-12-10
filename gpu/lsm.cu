// lsm.cu
#include "lsm.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector> 

// Error checking 

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
constexpr int MAX_KDIM = 8; 

inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " at " << file << ":" << line << std::endl;
        if (abort) std::exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

// Device helpers

__device__ inline double d_payoff(bool is_call, double K, double S) {
    if (is_call) return fmax(S - K, 0.0);
    else         return fmax(K - S, 0.0);
}

// polynomial basis: [1, S, S^2, ..., S^deg]
__device__ inline void d_eval_basis(double S, int deg, double* phi_out) {
    double power = 1.0;
    for (int k = 0; k <= deg; ++k) {
        phi_out[k] = power;
        power *= S;
    }
}

// simulate GBM paths
//
// d_paths: size = num_paths * (num_steps + 1)
// layout: paths[i*(M+1) + j]
//
__global__
void simulate_paths_kernel(double* __restrict__ d_paths,
                           std::size_t num_paths,
                           std::size_t num_steps,
                           double S0,
                           double r,
                           double sigma,
                           double T,
                           unsigned long long seed)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_paths) return;

    double dt = T / static_cast<double>(num_steps);
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrt_dt = sigma * sqrt(dt);

    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*sequence=*/ i, /*offset=*/ 0, &state);

    std::size_t stride = num_steps + 1;
    d_paths[i * stride + 0] = S0;

    for (std::size_t j = 1; j <= num_steps; ++j) {
        double z = curand_normal_double(&state);
        double log_growth = drift + vol_sqrt_dt * z;
        double prev = d_paths[i * stride + (j - 1)];
        d_paths[i * stride + j] = prev * exp(log_growth);
    }
}

// build regression A, b
//
// For each alive & ITM path at time j:
//
//   Y_i   = cashflow[i] * exp(-r * dt * (exercise_idx[i] - j))
//   phi   = basis(S_ij)
//   A    += phi * phi^T
//   b    += phi * Y_i
//
// We use atomicAdd since K is small (e.g., 3).
//
__global__
void regression_contribs_kernel(const double* __restrict__ d_paths,
                                const double* __restrict__ d_cashflow,
                                const int*    __restrict__ d_ex_idx,
                                double*       __restrict__ d_A,
                                double*       __restrict__ d_b,
                                std::size_t   num_paths,
                                std::size_t   num_steps,
                                int           deg,
                                double        r,
                                double        K,
                                bool          is_call,
                                double        dt,
                                int           j)
{
    const int Kdim = deg + 1;
    if (Kdim > MAX_KDIM) {
        return;
    }

    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid  = blockIdx.x;

    // Shared accumulators for this block
    extern __shared__ double sdata[]; 
    double* A_block = sdata;
    double* b_block = A_block + Kdim * Kdim; 

    for (int idx = tid; idx < Kdim * Kdim; idx += bdim) {
        A_block[idx] = 0.0;
    }
    for (int idx = tid; idx < Kdim; idx += bdim) {
        b_block[idx] = 0.0;
    }
    __syncthreads();

    std::size_t i = (std::size_t)bid * bdim + tid;
    if (i < num_paths) {
        int ex_idx = d_ex_idx[i];
        if (ex_idx > j) {
            int stride = (int)num_steps + 1;
            double S = d_paths[i * stride + j];
            double ex_pay = d_payoff(is_call, K, S);
            if (ex_pay > 0.0) {
                double tau = (double)(ex_idx - j);
                double Y   = d_cashflow[i] * exp(-r * dt * tau);

                double phi[MAX_KDIM];
                d_eval_basis(S, deg, phi);

                // atomically accumulate into block-local A_block / b_block
                for (int r_idx = 0; r_idx < Kdim; ++r_idx) {
                    for (int c_idx = 0; c_idx < Kdim; ++c_idx) {
                        double val = phi[r_idx] * phi[c_idx];
                        atomicAdd(&A_block[r_idx * Kdim + c_idx], val);
                    }
                    atomicAdd(&b_block[r_idx], phi[r_idx] * Y);
                }
            }
        }
    }

    __syncthreads();

    // One thread per block flushes block totals to global A, b
    if (tid == 0) {
        for (int idx = 0; idx < Kdim * Kdim; ++idx) {
            double v = A_block[idx];
            if (v != 0.0) {
                atomicAdd(&d_A[idx], v);
            }
        }
        for (int idx = 0; idx < Kdim; ++idx) {
            double v = b_block[idx];
            if (v != 0.0) {
                atomicAdd(&d_b[idx], v);
            }
        }
    }
}
// Kernel: apply exercise decision
//
// Using β (continuation coefficients) computed on host,
// compare immediate payoff vs estimated continuation.
//
__global__
void apply_exercise_kernel(const double* __restrict__ d_paths,
                           double*       __restrict__ d_cashflow,
                           int*          __restrict__ d_ex_idx,
                           const double* __restrict__ d_beta,
                           std::size_t num_paths,
                           std::size_t num_steps,
                           int   deg,
                           double K,
                           bool  is_call,
                           int   j)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_paths) return;

    int ex_id = d_ex_idx[i];
    if (ex_id <= j) return;  // already exercised

    int stride = (int)num_steps + 1;
    double S = d_paths[i * stride + j];
    double ex_pay = d_payoff(is_call, K, S);
    if (ex_pay <= 0.0) return;

    const int Kdim = deg + 1;
    double phi[8];
    d_eval_basis(S, deg, phi);

    double cont = 0.0;
    for (int k = 0; k < Kdim; ++k) {
        cont += d_beta[k] * phi[k];
    }

    if (ex_pay >= cont) {
        d_cashflow[i] = ex_pay;
        d_ex_idx[i]   = j;
    }
}

// Kernel: final discounted payoff reduction
//
// sum = Σ_i cashflow[i] * exp(-r * t_i),  t_i = ex_idx[i] * dt
//
__global__
void final_price_kernel(const double* __restrict__ d_cashflow,
                        const int*    __restrict__ d_ex_idx,
                        std::size_t   num_paths,
                        double r,
                        double dt,
                        double* d_sum_out)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    std::size_t i = blockIdx.x * blockDim.x + tid;

    double local = 0.0;
    for (; i < num_paths; i += blockDim.x * gridDim.x) {
        int ex_idx = d_ex_idx[i];
        double t   = dt * (double)ex_idx;
        double cf  = d_cashflow[i] * exp(-r * t);
        local += cf;
    }
    sdata[tid] = local;
    __syncthreads();

    // block reduction
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) sdata[tid] += sdata[tid + off];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(d_sum_out, sdata[0]);
}

// At maturity j = M, cashflow[i] = payoff(S_T), exercise_idx[i] = M.

__global__
void init_terminal_payoff_kernel(double*       __restrict__ d_cashflow,
                                 int*          __restrict__ d_ex_idx,
                                 const double* __restrict__ d_paths,
                                 std::size_t num_paths,
                                 std::size_t num_steps,
                                 double K,
                                 bool   is_call)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_paths) return;

    std::size_t stride = num_steps + 1;
    double S_T = d_paths[i * stride + num_steps];
    double pay = d_payoff(is_call, K, S_T);
    d_cashflow[i] = pay;
    d_ex_idx[i]   = (int)num_steps;
}



static void solve_small_linear(double* A, double* b, int n)
{
    for (int i = 0; i < n; ++i) {
        // pivot
        int piv = i;
        double maxv = std::fabs(A[i*n + i]);
        for (int r = i+1; r < n; ++r) {
            double v = std::fabs(A[r*n + i]);
            if (v > maxv) { maxv = v; piv = r; }
        }
        if (maxv < 1e-14) return; // singular-ish

        if (piv != i) {
            for (int c = 0; c < n; ++c) std::swap(A[i*n + c], A[piv*n + c]);
            std::swap(b[i], b[piv]);
        }

        double diag = A[i*n + i];
        for (int r = i+1; r < n; ++r) {
            double f = A[r*n + i] / diag;
            for (int c = i; c < n; ++c) {
                A[r*n + c] -= f * A[i*n + c];
            }
            b[r] -= f * b[i];
        }
    }

    // back-sub
    for (int i = n-1; i >= 0; --i) {
        double sum = b[i];
        for (int c = i+1; c < n; ++c) sum -= A[i*n + c] * b[c];
        double diag = A[i*n + i];
        if (std::fabs(diag) < 1e-14) continue;
        b[i] = sum / diag;
    }
}


double price_american_lsm_gpu(const ModelParams& model,
                              const OptionParams& opt,
                              const LsmConfig&    cfg,
                              double* total_ms,
                              double* kernel_ms)
{
    const std::size_t N = cfg.num_paths;
    const std::size_t M = cfg.num_steps;
    const int deg   = (int)cfg.poly_degree;
    const int Kdim  = deg + 1;

    double dt = opt.T / (double)M;

    // Allocate device arrays
    double* d_paths        = nullptr;
    double* d_cashflow     = nullptr;
    int*    d_ex_idx       = nullptr;
    double* d_A            = nullptr;
    double* d_b            = nullptr;
    double* d_beta         = nullptr;
    double* d_sum          = nullptr;

    std::size_t pathsBytes = sizeof(double) * N * (M + 1);
    std::size_t cfBytes    = sizeof(double) * N;
    std::size_t idxBytes   = sizeof(int)    * N;
    std::size_t ABytes     = sizeof(double) * Kdim * Kdim;
    std::size_t bBytes     = sizeof(double) * Kdim;

    cudaCheckError(cudaMalloc(&d_paths,    pathsBytes));
    cudaCheckError(cudaMalloc(&d_cashflow, cfBytes));
    cudaCheckError(cudaMalloc(&d_ex_idx,   idxBytes));
    cudaCheckError(cudaMalloc(&d_A,        ABytes));
    cudaCheckError(cudaMalloc(&d_b,        bBytes));
    cudaCheckError(cudaMalloc(&d_beta,     bBytes));
    cudaCheckError(cudaMalloc(&d_sum,      sizeof(double)));

    cudaCheckError(cudaMemset(d_sum, 0, sizeof(double)));

    std::vector<double> h_A(Kdim*Kdim);
    std::vector<double> h_b(Kdim);

    // CUDA events for timing
    cudaEvent_t ev_start, ev_end, ev_kstart, ev_kend;
    cudaCheckError(cudaEventCreate(&ev_start));
    cudaCheckError(cudaEventCreate(&ev_end));
    cudaCheckError(cudaEventCreate(&ev_kstart));
    cudaCheckError(cudaEventCreate(&ev_kend));

    cudaCheckError(cudaEventRecord(ev_start));

    const int blockSize = 256;
    const int gridPaths = (int)((N + blockSize - 1) / blockSize);

    // --- 1) Simulate paths ---
    cudaCheckError(cudaEventRecord(ev_kstart));
    simulate_paths_kernel<<<gridPaths, blockSize>>>(
        d_paths, N, M,
        model.S0, model.r, model.sigma,
        opt.T, cfg.rng_seed);
    cudaCheckError(cudaEventRecord(ev_kend));
    cudaCheckError(cudaEventSynchronize(ev_kend));

    float ms_kernel_total = 0.0f;
    cudaCheckError(cudaEventElapsedTime(&ms_kernel_total, ev_kstart, ev_kend));

    // 2) Initialize cashflow & exercise index at maturity on GPU
    init_terminal_payoff_kernel<<<gridPaths, blockSize>>>(
        d_cashflow, d_ex_idx,
        d_paths, N, M,
        opt.K, opt.is_call);
    cudaCheckError(cudaDeviceSynchronize());

    // 3) Backward induction over time steps
    for (int j = (int)M - 1; j >= 1; --j) {
        cudaCheckError(cudaMemset(d_A, 0, ABytes));
        cudaCheckError(cudaMemset(d_b, 0, bBytes));
        std::size_t shmem_reg = (Kdim * Kdim + Kdim) * sizeof(double);

        regression_contribs_kernel<<<gridPaths, blockSize, shmem_reg>>>(
            d_paths, d_cashflow, d_ex_idx,
            d_A, d_b,
            N, M,
            deg, model.r, opt.K, opt.is_call,
            dt, j);
        cudaCheckError(cudaDeviceSynchronize());

        // copy A, b back to host and solve A beta = b
        cudaCheckError(cudaMemcpy(h_A.data(), d_A, ABytes, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(h_b.data(), d_b, bBytes, cudaMemcpyDeviceToHost));

        solve_small_linear(h_A.data(), h_b.data(), Kdim);
        // h_b now holds β

        cudaCheckError(cudaMemcpy(d_beta, h_b.data(), bBytes, cudaMemcpyHostToDevice));

        // apply exercise decision on GPU
        apply_exercise_kernel<<<gridPaths, blockSize>>>(
            d_paths, d_cashflow, d_ex_idx,
            d_beta,
            N, M,
            deg, opt.K, opt.is_call,
            j);
        cudaCheckError(cudaDeviceSynchronize());
    }

    int gridRed = 120;
    std::size_t shmem = blockSize * sizeof(double);
    final_price_kernel<<<gridRed, blockSize, shmem>>>(
        d_cashflow, d_ex_idx,
        N, model.r, dt,
        d_sum);
    cudaCheckError(cudaDeviceSynchronize());

    double h_sum = 0.0;
    cudaCheckError(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    double price = h_sum / (double)N;

    cudaCheckError(cudaEventRecord(ev_end));
    cudaCheckError(cudaEventSynchronize(ev_end));

    float ms_total = 0.0f;
    cudaCheckError(cudaEventElapsedTime(&ms_total, ev_start, ev_end));

    if (total_ms)  *total_ms  = (double)ms_total;
    if (kernel_ms) *kernel_ms = (double)ms_kernel_total;

    // cleanup
    cudaCheckError(cudaFree(d_paths));
    cudaCheckError(cudaFree(d_cashflow));
    cudaCheckError(cudaFree(d_ex_idx));
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_b));
    cudaCheckError(cudaFree(d_beta));
    cudaCheckError(cudaFree(d_sum));

    cudaCheckError(cudaEventDestroy(ev_start));
    cudaCheckError(cudaEventDestroy(ev_end));
    cudaCheckError(cudaEventDestroy(ev_kstart));
    cudaCheckError(cudaEventDestroy(ev_kend));

    return price;
}
