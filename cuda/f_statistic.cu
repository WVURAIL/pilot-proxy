/**
 * @file f_statistic.cu
 * @brief GPU-accelerated F-statistic computation for narrowband signal detection.
 *
 * Implements a matched-filter F-statistic detector optimized for detecting
 * DTV pilot tones in wideband spectral data. Uses fixed-point arithmetic
 * through the dot-product and power-accumulation stages.
 *
 * @section algorithm Algorithm Overview
 *
 * For each detector row and weight term:
 *   1. Compute complex dot product: z[row,term] = sum_k x[m,k] * conj(w[n,k])
 *   2. Accumulate power: P[term] = sum_row |z[row,term]|^2
 *
 * The statistic is a target/reference power ratio:
 *
 *   F = 2 * P[target] / (P[ref1] + P[ref2])
 *
 * Thresholds are supplied by the external calibration/science-tolerance chain
 * and are applied as raw F-statistic thresholds.
 *
 * @section optimization Optimization Strategy
 *
 * - Integer arithmetic in the detector-window loop (avoids float conversion overhead)
 * - Warp shuffle reductions (no shared memory bank conflicts)
 * - Grid-stride loop over detector rows
 * - Weight ROM stores natural matched-filter weights
 * - Packed weights uploaded once per call; optional DP4A path pre-packs lanes
 * - uint64 power accumulation over detector rows (float only at final division/output)
 *
 * @author Dylan
 * @date 2025
 */

#include "config.h"
#include "f_statistic.h"

#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstring>
#include <new>

#define FSTAT_WEIGHT_COUNT (FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES)
#define FSTAT_WEIGHT_BYTES (FSTAT_WEIGHT_COUNT * sizeof(InputType))
#define FSTAT_WEIGHT_LANE_COUNT (FSTAT_NUM_WEIGHT_TERMS * FSTAT_DP4A_TAP_PAIRS)

#define FSTAT_CUDA_MAX_THREADS_PER_BLOCK 1024
#define FSTAT_OUTPUT_KERNEL_THREADS 256
#define FSTAT_DP4A_LANE_BITS 8
#define FSTAT_DP4A_LANE1_SHIFT FSTAT_DP4A_LANE_BITS
#define FSTAT_DP4A_LANE2_SHIFT (2 * FSTAT_DP4A_LANE_BITS)
#define FSTAT_DP4A_LANE3_SHIFT (3 * FSTAT_DP4A_LANE_BITS)
#define FSTAT_INT4_MASK ((1 << FSTAT_INT4_COMPONENT_BITS) - 1)
#define FSTAT_INT4_SIGN_BIT (1 << (FSTAT_INT4_COMPONENT_BITS - 1))

static_assert(
    FSTAT_BLOCK_THREADS >= FSTAT_WARP_SIZE,
    "FSTAT_BLOCK_THREADS must be at least one CUDA warp.");
static_assert(
    (FSTAT_BLOCK_THREADS % FSTAT_WARP_SIZE) == 0,
    "FSTAT_BLOCK_THREADS must be a multiple of FSTAT_WARP_SIZE.");
static_assert(
    FSTAT_BLOCK_THREADS <= FSTAT_CUDA_MAX_THREADS_PER_BLOCK,
    "FSTAT_BLOCK_THREADS exceeds the CUDA per-block thread limit.");
static_assert(
    FSTAT_WARPS_PER_BLOCK <= FSTAT_WARP_SIZE,
    "Block reduction assumes at most 32 warps per block.");

#if FSTAT_USE_DP4A
static_assert(
    FSTAT_SAMPLE_BITS_PER_COMPONENT == 4,
    "DP4A path assumes packed complex int4+int4 samples.");
static_assert(
    (FSTAT_DETECTOR_WINDOW_SAMPLES % 2) == 0,
    "DP4A path requires an even detector window.");
#endif

#if FSTAT_USE_DP4A && FSTAT_USE_CONSTANT_WEIGHT_LANES
__constant__ int c_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
#endif

/* ===========================================================================
 * CUDA ERROR CHECKING
 * ===========================================================================*/

static thread_local char g_last_error[512] = "";

static void clear_last_error()
{
    g_last_error[0] = '\0';
}

static void record_api_error(const char* message)
{
    std::snprintf(
        g_last_error,
        sizeof(g_last_error),
        "FStat API error: %s",
        message);
    std::fprintf(stderr, "%s\n", g_last_error);
}

static bool record_cuda_error(cudaError_t err, const char* file, int line)
{
    if (err == cudaSuccess) {
        return true;
    }
    std::snprintf(
        g_last_error,
        sizeof(g_last_error),
        "CUDA error %s:%d: %s",
        file,
        line,
        cudaGetErrorString(err));
    std::fprintf(stderr, "%s\n", g_last_error);
    return false;
}

#define CUDA_CHECK(call) do { \
    if (!record_cuda_error((call), __FILE__, __LINE__)) { \
        return; \
    } \
} while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#define CUDA_CHECK_BOOL(call) do { \
    if (!record_cuda_error((call), __FILE__, __LINE__)) { \
        return false; \
    } \
} while (0)

#define CUDA_CHECK_LAST_BOOL() CUDA_CHECK_BOOL(cudaGetLastError())

#ifndef NDEBUG
#define CUDA_CHECK_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
#define CUDA_CHECK_SYNC_BOOL() CUDA_CHECK_BOOL(cudaDeviceSynchronize())
#else
#define CUDA_CHECK_SYNC() do { } while (0)
#define CUDA_CHECK_SYNC_BOOL() do { } while (0)
#endif

/* ===========================================================================
 * DEVICE HELPER FUNCTIONS
 * ===========================================================================*/

/**
 * @brief Sign-extend a packed n-bit two's-complement component.
 */
__device__ __forceinline__
int sign_extend_nbits(int x, int bits)
{
    const int mask = (1 << bits) - 1;
    const int sign = 1 << (bits - 1);

    x &= mask;
    return (x ^ sign) - sign;
}

/**
 * @brief Unpack a packed complex sample to real/imag components.
 *
 * Handles sign extension for both components:
 *   - Real: Upper FSTAT_SAMPLE_BITS_PER_COMPONENT bits
 *   - Imag: Lower FSTAT_SAMPLE_BITS_PER_COMPONENT bits
 *
 * @param packed  Packed complex value
 * @return short2 with .x = real, .y = imag
 */
__device__ __forceinline__
short2 unpack_sample(InputType packed)
{
    constexpr int bits = FSTAT_SAMPLE_BITS_PER_COMPONENT;
    constexpr int mask = (1 << bits) - 1;

    const int byte = static_cast<int>(static_cast<unsigned char>(packed));

    const int real = sign_extend_nbits(byte >> bits, bits);
    const int imag = sign_extend_nbits(byte & mask, bits);

    return make_short2(
        static_cast<short>(real),
        static_cast<short>(imag));
}

/**
 * @brief Complex multiply by the conjugate of a matched-filter weight.
 *
 * Computes x times conj(w) using integer arithmetic.
 * Result uses 32-bit integers to hold 16x16 products without overflow.
 *
 * @param x  Input sample (real, imag)
 * @param w  Natural matched-filter weight
 * @return int2 with .x = real part, .y = imag part
 */
__device__ __forceinline__
int2 complex_multiply(short2 x, short2 w)
{
    // (x.r + j*x.i) x conj(w.r + j*w.i)
    // = (x.r*w.r + x.i*w.i) + j*(x.i*w.r - x.r*w.i)
    return make_int2(
        static_cast<int>(x.x) * w.x + static_cast<int>(x.y) * w.y,  // Real
        static_cast<int>(x.y) * w.x - static_cast<int>(x.x) * w.y   // Imag
    );
}

/**
 * @brief Complex addition (integer).
 */
__device__ __forceinline__
int2 complex_add(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

#if FSTAT_USE_DP4A
/**
 * @brief Sign-extend a packed 4-bit two's-complement component.
 */
__device__ __forceinline__
int sign_extend_i4(int x)
{
    return sign_extend_nbits(x, FSTAT_INT4_COMPONENT_BITS);
}

/**
 * @brief Pack four signed int8 lanes into the 32-bit format consumed by DP4A.
 */
__device__ __forceinline__
int pack4_i8(int a0, int a1, int a2, int a3)
{
    unsigned int u0 = static_cast<unsigned char>(static_cast<signed char>(a0));
    unsigned int u1 = static_cast<unsigned char>(static_cast<signed char>(a1));
    unsigned int u2 = static_cast<unsigned char>(static_cast<signed char>(a2));
    unsigned int u3 = static_cast<unsigned char>(static_cast<signed char>(a3));
    return static_cast<int>(
        u0
        | (u1 << FSTAT_DP4A_LANE1_SHIFT)
        | (u2 << FSTAT_DP4A_LANE2_SHIFT)
        | (u3 << FSTAT_DP4A_LANE3_SHIFT));
}

/**
 * @brief Decode two packed complex samples into DP4A lanes.
 *
 * For x * conj(w):
 *   real = [xr0, xi0, xr1, xi1] dot [wr0, wi0, wr1, wi1]
 *   imag = [xi0,-xr0, xi1,-xr1] dot [wr0, wi0, wr1, wi1]
 *
 * The DP4A lanes are int8 containers. They contain signed int4 values
 * and, for the imaginary lane, negated signed int4 values. If a data
 * component is -8, its negation is +8, which is representable in int8.
 * The numerical product bound is still governed by the int4 container
 * magnitude, so the 4 + 4 + 1 + log2(128) = 16 bit-growth argument holds.
 */
__device__ __forceinline__
void unpack_two_complex_bytes_to_dp4a_lanes(
    InputType x0,
    InputType x1,
    int& a_re,
    int& a_im)
{
    int b0 = static_cast<int>(static_cast<unsigned char>(x0));
    int b1 = static_cast<int>(static_cast<unsigned char>(x1));

    int x0r = sign_extend_i4(b0 >> FSTAT_INT4_COMPONENT_BITS);
    int x0i = sign_extend_i4(b0);
    int x1r = sign_extend_i4(b1 >> FSTAT_INT4_COMPONENT_BITS);
    int x1i = sign_extend_i4(b1);

    a_re = pack4_i8(x0r, x0i, x1r, x1i);
    a_im = pack4_i8(x0i, -x0r, x1i, -x1r);
}

/**
 * @brief Load a prepacked DP4A weight lane.
 *
 * The DP4A path can load the tiny 768-byte lane table from CUDA constant
 * memory or from a per-handle global buffer preloaded into shared memory. The
 * default production configuration uses shared-memory preload because
 * different threads in a warp read different pair indices.
 */
__device__ __forceinline__
int load_weight_lane(const int* W_Lanes, int idx)
{
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
    (void)W_Lanes;
    return c_weight_lanes[idx];
    #else
    return W_Lanes[idx];
    #endif
}
#endif

/**
 * @brief Warp-level sum reduction for int2.
 *
 * Uses shuffle intrinsics for fast intra-warp communication.
 * Result is valid only in lane 0 after completion.
 *
 * @param val  Per-thread value to reduce
 * @return Warp sum (valid in lane 0 only)
 */
__device__ __forceinline__
int2 warp_reduce_sum(int2 val)
{
    #pragma unroll
    for (int offset = FSTAT_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(FSTAT_WARP_MASK, val.x, offset);
        val.y += __shfl_down_sync(FSTAT_WARP_MASK, val.y, offset);
    }
    return val;
}

#ifndef NDEBUG
__device__ __forceinline__
void debug_check_dot_int16_bounds(
    const int2* dot,
    int detector_row,
    int batch_index)
{
    #pragma unroll
    for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
        if (dot[n].x < SHRT_MIN || dot[n].x > SHRT_MAX ||
            dot[n].y < SHRT_MIN || dot[n].y > SHRT_MAX) {
            if (batch_index >= 0) {
                printf(
                    "FStat dot product exceeded int16 bound: batch=%d row=%d term=%d real=%d imag=%d\n",
                    batch_index,
                    detector_row,
                    n,
                    dot[n].x,
                    dot[n].y);
            } else {
                printf(
                    "FStat dot product exceeded int16 bound: row=%d term=%d real=%d imag=%d\n",
                    detector_row,
                    n,
                    dot[n].x,
                    dot[n].y);
            }
        }
    }
}
#endif

/* ===========================================================================
 * CUDA KERNELS
 * ===========================================================================*/

/**
 * @brief Main power accumulation kernel.
 *
 * Computes power in each weight vector's matched filter output,
 * accumulated over all detector rows in one block.
 *
 * Algorithm per detector block:
 *   1. Grid-stride loop over detector rows
 *   2. Thread-stride loop over detector-window samples (complex dot product)
 *   3. Warp reduction -> shared memory -> block reduction
 *   4. Accumulate |dot_product|^2 into per-block uint64 accumulator
 *   5. Atomic add block results to global integer power scratch
 *
 * @param X            Row-major input samples [detector_rows_per_block x detector_window_samples],
 *                     packed complex, with X[m, k] = X[m * detector_window_samples + k]
 * @param Power_Terms  Output power per weight vector [num_weight_terms]
 * @param detector_rows_per_block  Number of detector rows in the matrix view
 */
__global__ void
kernel_accumulate_power(
    const InputType* __restrict__ X,
    const InputType* __restrict__ W,
    const int*       __restrict__ W_Lanes,
    unsigned long long* __restrict__ Power_Terms,
    int detector_rows_per_block)
{
    #if FSTAT_USE_DP4A
    (void)W;
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
    (void)W_Lanes;
    #endif
    #else
    (void)W_Lanes;
    #endif

    // Shared memory for inter-warp reduction
    __shared__ int2 warp_sums[FSTAT_NUM_WEIGHT_TERMS][FSTAT_WARPS_PER_BLOCK];
    __shared__ unsigned long long block_power[FSTAT_NUM_WEIGHT_TERMS];

    const int tid     = threadIdx.x;
    const int warp_id = tid / FSTAT_WARP_SIZE;
    const int lane_id = tid % FSTAT_WARP_SIZE;

    if (tid < FSTAT_NUM_WEIGHT_TERMS) {
        block_power[tid] = 0ULL;
    }
    __syncthreads();

    #if FSTAT_USE_DP4A && FSTAT_USE_SHARED_WEIGHT_LANES && !FSTAT_USE_CONSTANT_WEIGHT_LANES
    const int* weight_lanes = W_Lanes;
    __shared__ int shared_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
    for (int idx = tid; idx < FSTAT_WEIGHT_LANE_COUNT; idx += blockDim.x) {
        shared_weight_lanes[idx] = W_Lanes[idx];
    }
    __syncthreads();
    weight_lanes = shared_weight_lanes;
    #elif FSTAT_USE_DP4A
    const int* weight_lanes = W_Lanes;
    #endif

    // -------------------------------------------------------------------------
    // Grid-stride loop over detector rows
    // -------------------------------------------------------------------------
    for (int m = blockIdx.x; m < detector_rows_per_block; m += gridDim.x) {

        // Per-thread dot product accumulators
        int2 dot[FSTAT_NUM_WEIGHT_TERMS];
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = make_int2(0, 0);
        }

        // ---------------------------------------------------------------------
        // Thread-stride loop over frequency taps
        // ---------------------------------------------------------------------
        #if FSTAT_USE_DP4A
        for (int pair = tid; pair < FSTAT_DP4A_TAP_PAIRS; pair += blockDim.x) {
            const int k0 = 2 * pair;
            const int k1 = k0 + 1;

            int a_re;
            int a_im;
            unpack_two_complex_bytes_to_dp4a_lanes(
                X[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k0],
                X[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k1],
                a_re,
                a_im);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                const int w_lane =
                    load_weight_lane(weight_lanes, n * FSTAT_DP4A_TAP_PAIRS + pair);
                dot[n].x = __dp4a(a_re, w_lane, dot[n].x);
                dot[n].y = __dp4a(a_im, w_lane, dot[n].y);
            }
        }
        #else
        for (int k = tid; k < FSTAT_DETECTOR_WINDOW_SAMPLES; k += blockDim.x) {
            short2 x_val = unpack_sample(X[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                short2 w_val = unpack_sample(W[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);
                dot[n] = complex_add(dot[n], complex_multiply(x_val, w_val));
            }
        }
        #endif

        // ---------------------------------------------------------------------
        // Warp reduction
        // ---------------------------------------------------------------------
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = warp_reduce_sum(dot[n]);
        }

        // Write warp results to shared memory
        if (lane_id == 0) {
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                warp_sums[n][warp_id] = dot[n];
            }
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // Block reduction (warp 0 only)
        // ---------------------------------------------------------------------
        if (warp_id == 0) {
            // Load from shared memory
            if (lane_id < FSTAT_WARPS_PER_BLOCK) {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = warp_sums[n][lane_id];
                }
            } else {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = make_int2(0, 0);
                }
            }

            // Final warp reduction
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                dot[n] = warp_reduce_sum(dot[n]);
            }

            // Thread 0: compute magnitude squared and accumulate
            if (tid == 0) {
                #ifndef NDEBUG
                debug_check_dot_int16_bounds(dot, m, -1);
                #endif

                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    // Use 64-bit to prevent overflow in |z|^2 computation
                    long long zr = static_cast<long long>(dot[n].x);
                    long long zi = static_cast<long long>(dot[n].y);
                    unsigned long long mag_sq =
                        static_cast<unsigned long long>(zr * zr + zi * zi);

                    block_power[n] += mag_sq;
                }
            }
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Write block results to global memory
    // -------------------------------------------------------------------------
    if (tid == 0) {
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            atomicAdd(&Power_Terms[n], block_power[n]);
        }
    }
}

/**
 * @brief Batched power accumulation kernel.
 *
 * Processes `batch` independent row-major input blocks, each of shape
 * [detector_rows_per_block x detector_window_samples].
 * The batch index is mapped to blockIdx.y.
 *
 * @param X            Row-major input samples [batch x detector_rows_per_block x detector_window_samples],
 *                     packed complex
 * @param Power_Terms  Output power per weight vector [batch x num_weight_terms]
 * @param detector_rows_per_block  Number of detector rows in the matrix view per block
 * @param batch        Number of independent blocks
 */
__global__ void
kernel_accumulate_power_batched(
    const InputType* __restrict__ X,
    const InputType* __restrict__ W,
    const int*       __restrict__ W_Lanes,
    unsigned long long* __restrict__ Power_Terms,
    int detector_rows_per_block,
    int batch)
{
    #if FSTAT_USE_DP4A
    (void)W;
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
    (void)W_Lanes;
    #endif
    #else
    (void)W_Lanes;
    #endif

    const int b = blockIdx.y;
    if (b >= batch) {
        return;
    }

    const size_t batch_stride = static_cast<size_t>(detector_rows_per_block) * FSTAT_DETECTOR_WINDOW_SAMPLES;
    const InputType* Xb = X + batch_stride * static_cast<size_t>(b);
    unsigned long long* P = Power_Terms + (b * FSTAT_NUM_WEIGHT_TERMS);

    // Shared memory for inter-warp reduction
    __shared__ int2 warp_sums[FSTAT_NUM_WEIGHT_TERMS][FSTAT_WARPS_PER_BLOCK];
    __shared__ unsigned long long block_power[FSTAT_NUM_WEIGHT_TERMS];

    const int tid     = threadIdx.x;
    const int warp_id = tid / FSTAT_WARP_SIZE;
    const int lane_id = tid % FSTAT_WARP_SIZE;

    if (tid < FSTAT_NUM_WEIGHT_TERMS) {
        block_power[tid] = 0ULL;
    }
    __syncthreads();

    #if FSTAT_USE_DP4A && FSTAT_USE_SHARED_WEIGHT_LANES && !FSTAT_USE_CONSTANT_WEIGHT_LANES
    const int* weight_lanes = W_Lanes;
    __shared__ int shared_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
    for (int idx = tid; idx < FSTAT_WEIGHT_LANE_COUNT; idx += blockDim.x) {
        shared_weight_lanes[idx] = W_Lanes[idx];
    }
    __syncthreads();
    weight_lanes = shared_weight_lanes;
    #elif FSTAT_USE_DP4A
    const int* weight_lanes = W_Lanes;
    #endif

    for (int m = blockIdx.x; m < detector_rows_per_block; m += gridDim.x) {
        int2 dot[FSTAT_NUM_WEIGHT_TERMS];
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = make_int2(0, 0);
        }

        #if FSTAT_USE_DP4A
        for (int pair = tid; pair < FSTAT_DP4A_TAP_PAIRS; pair += blockDim.x) {
            const int k0 = 2 * pair;
            const int k1 = k0 + 1;

            int a_re;
            int a_im;
            unpack_two_complex_bytes_to_dp4a_lanes(
                Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k0],
                Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k1],
                a_re,
                a_im);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                const int w_lane =
                    load_weight_lane(weight_lanes, n * FSTAT_DP4A_TAP_PAIRS + pair);
                dot[n].x = __dp4a(a_re, w_lane, dot[n].x);
                dot[n].y = __dp4a(a_im, w_lane, dot[n].y);
            }
        }
        #else
        for (int k = tid; k < FSTAT_DETECTOR_WINDOW_SAMPLES; k += blockDim.x) {
            short2 x_val = unpack_sample(Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                short2 w_val = unpack_sample(W[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);
                dot[n] = complex_add(dot[n], complex_multiply(x_val, w_val));
            }
        }
        #endif

        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = warp_reduce_sum(dot[n]);
        }

        if (lane_id == 0) {
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                warp_sums[n][warp_id] = dot[n];
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < FSTAT_WARPS_PER_BLOCK) {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = warp_sums[n][lane_id];
                }
            } else {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = make_int2(0, 0);
                }
            }

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                dot[n] = warp_reduce_sum(dot[n]);
            }

            if (tid == 0) {
                #ifndef NDEBUG
                debug_check_dot_int16_bounds(dot, m, b);
                #endif

                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    long long zr = static_cast<long long>(dot[n].x);
                    long long zi = static_cast<long long>(dot[n].y);
                    unsigned long long mag_sq =
                        static_cast<unsigned long long>(zr * zr + zi * zi);
                    block_power[n] += mag_sq;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            atomicAdd(&P[n], block_power[n]);
        }
    }
}

/**
 * @brief Compute F-statistic from accumulated powers.
 *
 * F = 2 x P[target] / (P[ref1] + P[ref2])
 *
 * @param P         Power terms [num_weight_terms] from accumulation kernel
 * @param F_Result  Output F-statistic (single value)
 */
__global__ void
kernel_compute_f_statistic(
    const unsigned long long* __restrict__ P,
    float*       __restrict__ F_Result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double denominator =
            static_cast<double>(P[FSTAT_LOWER_REFERENCE_WEIGHT_INDEX])
            + static_cast<double>(P[FSTAT_UPPER_REFERENCE_WEIGHT_INDEX]);

        if (denominator > 0.0) {
            double numerator =
                FSTAT_RAW_NUMDEN_SCALE
                * static_cast<double>(P[FSTAT_TARGET_WEIGHT_INDEX]);
            *F_Result = static_cast<float>(numerator / denominator);
        } else {
            *F_Result = 0.0f;  // Degenerate case: no reference power
        }
    }
}

/**
 * @brief Compute F-statistic for each batch entry.
 *
 * @param P         Power terms [batch x num_weight_terms] from accumulation kernel
 * @param F_Result  Output F-statistics [batch]
 * @param batch     Number of batch entries
 */
__global__ void
kernel_compute_f_statistic_batched(
    const unsigned long long* __restrict__ P,
    float*       __restrict__ F_Result,
    int batch)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }
    const unsigned long long* Pb = P + (b * FSTAT_NUM_WEIGHT_TERMS);
    double denominator =
        static_cast<double>(Pb[FSTAT_LOWER_REFERENCE_WEIGHT_INDEX])
        + static_cast<double>(Pb[FSTAT_UPPER_REFERENCE_WEIGHT_INDEX]);
    if (denominator > 0.0) {
        double numerator =
            FSTAT_RAW_NUMDEN_SCALE
            * static_cast<double>(Pb[FSTAT_TARGET_WEIGHT_INDEX]);
        F_Result[b] = static_cast<float>(numerator / denominator);
    } else {
        F_Result[b] = 0.0f;
    }
}

/**
 * @brief Convert integer power terms to a diagnostic float output buffer.
 */
__global__ void
kernel_convert_power_terms_to_float(
    const unsigned long long* __restrict__ P,
    float* __restrict__ Out,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    Out[idx] = static_cast<float>(P[idx]);
}

/**
 * @brief Saturating uint64 multiply for defensive rational-threshold products.
 *
 * The locked detector's shelf thresholds and power ranges are expected to fit
 * without saturation. Saturation is defensive only; any debug-build saturation
 * report should be treated as a validation failure for deployed thresholds.
 */
__device__ __forceinline__
void record_rational_overflow(unsigned int* overflow_count)
{
    if (overflow_count != nullptr) {
        atomicAdd(overflow_count, 1u);
    }
}

__device__ __forceinline__
unsigned long long saturating_mul_u64(
    unsigned long long a,
    unsigned long long b,
    unsigned int* overflow_count)
{
    constexpr unsigned long long max_u64 = ~0ULL;
    if (a != 0ULL && b > max_u64 / a) {
        record_rational_overflow(overflow_count);
        #ifndef NDEBUG
        printf(
            "FStat rational threshold multiply saturated: %llu * %llu\n",
            a,
            b);
        #endif
        return max_u64;
    }
    return a * b;
}

/**
 * @brief Saturating uint64 add for defensive reference-power sums.
 */
__device__ __forceinline__
unsigned long long saturating_add_u64(
    unsigned long long a,
    unsigned long long b,
    unsigned int* overflow_count)
{
    constexpr unsigned long long max_u64 = ~0ULL;
    if (b > max_u64 - a) {
        record_rational_overflow(overflow_count);
        #ifndef NDEBUG
        printf(
            "FStat rational threshold add saturated: %llu + %llu\n",
            a,
            b);
        #endif
        return max_u64;
    }
    return a + b;
}

/**
 * @brief Write P_target, P_ref1 + P_ref2, and half-threshold mask decisions.
 *
 * Applies, when P_ref1 + P_ref2 is nonzero:
 *
 *     P_target / (P_ref1 + P_ref2) >=
 *         threshold_half_num / threshold_half_den
 *
 * where threshold_half_num / threshold_half_den is one half of the full raw
 * F-statistic threshold. If P_ref1 + P_ref2 is zero, the reference power is
 * invalid for the deployed detector and the mask is forced to zero.
 */
__global__ void
kernel_write_num_den_mask_threshold_half_rational(
    const unsigned long long* __restrict__ P,
    unsigned long long* __restrict__ Num_Result,
    unsigned long long* __restrict__ Den_Result,
    unsigned char* __restrict__ Mask_Result,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    int batch,
    unsigned int* __restrict__ rational_overflow_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) {
        return;
    }

    const unsigned long long* Pb = P + (idx * FSTAT_NUM_WEIGHT_TERMS);
    const unsigned long long denominator = saturating_add_u64(
        Pb[FSTAT_LOWER_REFERENCE_WEIGHT_INDEX],
        Pb[FSTAT_UPPER_REFERENCE_WEIGHT_INDEX],
        rational_overflow_count);

    Num_Result[idx] = Pb[FSTAT_TARGET_WEIGHT_INDEX];
    Den_Result[idx] = denominator;

    if (threshold_half_den == 0ULL || denominator == 0ULL) {
        Mask_Result[idx] = 0u;
        return;
    }

    const unsigned long long lhs = saturating_mul_u64(
        Pb[FSTAT_TARGET_WEIGHT_INDEX],
        threshold_half_den,
        rational_overflow_count);
    const unsigned long long rhs = saturating_mul_u64(
        threshold_half_num,
        denominator,
        rational_overflow_count);
    Mask_Result[idx] = (lhs >= rhs) ? 1u : 0u;
}

/* ===========================================================================
 * INTERNAL DATA STRUCTURES
 * ===========================================================================*/

/**
 * @brief Opaque handle for F-statistic computation state.
 */
struct FStatHandle {
    int        detector_rows_per_block; ///< Rows in detector-matrix view
    int        batch;      ///< Batch size (number of independent blocks)
    const InputType* d_in; ///< Device input pointer (external)
    float*     d_out;      ///< Device output pointer (external)
    unsigned long long* d_power_scratch; ///< Internal integer power sums [num_weight_terms per block]
    InputType* d_weights;  ///< Scalar-path weights [num_weight_terms x detector_window_samples]
    int*       d_weight_lanes; ///< DP4A-path packed weight lanes [num_weight_terms x tap_pairs]
    bool       weights_cached; ///< Host cache validity for avoiding repeated uploads
    InputType  h_weight_cache[FSTAT_WEIGHT_COUNT]; ///< Last packed weights supplied by caller
};

#if FSTAT_USE_DP4A
static int host_sign_extend_i4(int x)
{
    x &= FSTAT_INT4_MASK;
    return (x ^ FSTAT_INT4_SIGN_BIT) - FSTAT_INT4_SIGN_BIT;
}

static int host_pack4_i8(int a0, int a1, int a2, int a3)
{
    unsigned int u0 = static_cast<unsigned char>(static_cast<signed char>(a0));
    unsigned int u1 = static_cast<unsigned char>(static_cast<signed char>(a1));
    unsigned int u2 = static_cast<unsigned char>(static_cast<signed char>(a2));
    unsigned int u3 = static_cast<unsigned char>(static_cast<signed char>(a3));
    return static_cast<int>(
        u0
        | (u1 << FSTAT_DP4A_LANE1_SHIFT)
        | (u2 << FSTAT_DP4A_LANE2_SHIFT)
        | (u3 << FSTAT_DP4A_LANE3_SHIFT));
}

static void host_unpack_i4_complex(InputType packed, int& real, int& imag)
{
    const int byte = static_cast<int>(static_cast<unsigned char>(packed));
    real = host_sign_extend_i4(byte >> FSTAT_INT4_COMPONENT_BITS);
    imag = host_sign_extend_i4(byte);
}

static void prepack_weight_lanes(
    const InputType* w_in,
    int* h_weight_lanes)
{
    for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
        for (int pair = 0; pair < FSTAT_DP4A_TAP_PAIRS; ++pair) {
            const int k0 = 2 * pair;
            const int k1 = k0 + 1;

            int w0r;
            int w0i;
            int w1r;
            int w1i;
            host_unpack_i4_complex(w_in[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k0], w0r, w0i);
            host_unpack_i4_complex(w_in[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k1], w1r, w1i);

            h_weight_lanes[n * FSTAT_DP4A_TAP_PAIRS + pair] =
                host_pack4_i8(w0r, w0i, w1r, w1i);
        }
    }
}
#endif

static bool fstat_accumulate(FStatHandle* h, const InputType* w_in)
{
    if (!h) {
        record_api_error("handle is null.");
        return false;
    }
    if (!w_in) {
        record_api_error("weight pointer is null.");
        return false;
    }

    #if FSTAT_USE_DP4A && FSTAT_USE_CONSTANT_WEIGHT_LANES
    // Constant memory is module-global, not per-handle. Always refresh it
    // before launches to avoid cross-handle stale weight-lane state.
    const bool weights_changed = true;
    #else
    const bool weights_changed =
        !h->weights_cached ||
        (std::memcmp(h->h_weight_cache, w_in, FSTAT_WEIGHT_BYTES) != 0);
    #endif

    if (weights_changed) {
        h->weights_cached = false;

    #if FSTAT_USE_DP4A
        int h_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
        prepack_weight_lanes(w_in, h_weight_lanes);
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
        CUDA_CHECK_BOOL(cudaMemcpyToSymbol(
            c_weight_lanes,
            h_weight_lanes,
            FSTAT_WEIGHT_LANE_COUNT * sizeof(int)));
    #else
        CUDA_CHECK_BOOL(cudaMemcpy(
            h->d_weight_lanes,
            h_weight_lanes,
            FSTAT_WEIGHT_LANE_COUNT * sizeof(int),
            cudaMemcpyHostToDevice));
    #endif
    #else
        CUDA_CHECK_BOOL(cudaMemcpy(
            h->d_weights,
            w_in,
            FSTAT_WEIGHT_BYTES,
            cudaMemcpyHostToDevice));
    #endif

        std::memcpy(h->h_weight_cache, w_in, FSTAT_WEIGHT_BYTES);
        h->weights_cached = true;
    }

    // Clear integer power scratch buffer
    CUDA_CHECK_BOOL(cudaMemset(
        h->d_power_scratch,
        0,
        h->batch * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long)));

    // Compute grid dimensions
    int grid_size = (h->detector_rows_per_block + FSTAT_BLOCK_THREADS - 1) / FSTAT_BLOCK_THREADS;
    if (grid_size > FSTAT_GRID_MAX_BLOCKS) {
        grid_size = FSTAT_GRID_MAX_BLOCKS;
    }

    if (h->batch <= 1) {
        kernel_accumulate_power<<<grid_size, FSTAT_BLOCK_THREADS>>>(
            h->d_in,
            h->d_weights,
            h->d_weight_lanes,
            h->d_power_scratch,
            h->detector_rows_per_block
        );
    } else {
        dim3 grid(grid_size, h->batch, 1);
        kernel_accumulate_power_batched<<<grid, FSTAT_BLOCK_THREADS>>>(
            h->d_in,
            h->d_weights,
            h->d_weight_lanes,
            h->d_power_scratch,
            h->detector_rows_per_block,
            h->batch
        );
    }
    CUDA_CHECK_LAST_BOOL();
    CUDA_CHECK_SYNC_BOOL();
    return true;
}

static bool fstat_write_f_statistic(FStatHandle* h)
{
    if (!h) {
        record_api_error("handle is null.");
        return false;
    }
    if (h->d_out == nullptr) {
        record_api_error("output pointer is null for floating F-statistic output.");
        return false;
    }

    if (h->batch <= 1) {
        kernel_compute_f_statistic<<<1, 1>>>(
            h->d_power_scratch, h->d_out
        );
    } else {
        const int threads = FSTAT_OUTPUT_KERNEL_THREADS;
        const int blocks = (h->batch + threads - 1) / threads;
        kernel_compute_f_statistic_batched<<<blocks, threads>>>(
            h->d_power_scratch, h->d_out, h->batch
        );
    }
    CUDA_CHECK_LAST_BOOL();
    CUDA_CHECK_SYNC_BOOL();
    return true;
}

static bool fstat_write_numden_mask_rational(
    FStatHandle* h,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    unsigned long long* d_num_out,
    unsigned long long* d_den_out,
    unsigned char* d_mask_out,
    unsigned int* d_rational_overflow_count)
{
    if (!h) {
        record_api_error("handle is null.");
        return false;
    }
    if (d_num_out == nullptr) {
        record_api_error("numerator output pointer is null.");
        return false;
    }
    if (d_den_out == nullptr) {
        record_api_error("denominator output pointer is null.");
        return false;
    }
    if (d_mask_out == nullptr) {
        record_api_error("mask output pointer is null.");
        return false;
    }
    if (threshold_half_den == 0ULL) {
        record_api_error("threshold_half_den must be nonzero.");
        return false;
    }
    if (d_rational_overflow_count != nullptr) {
        CUDA_CHECK_BOOL(cudaMemset(
            d_rational_overflow_count,
            0,
            sizeof(unsigned int)));
    }

    const int threads = FSTAT_OUTPUT_KERNEL_THREADS;
    const int blocks = (h->batch + threads - 1) / threads;
    kernel_write_num_den_mask_threshold_half_rational<<<blocks, threads>>>(
        h->d_power_scratch,
        d_num_out,
        d_den_out,
        d_mask_out,
        threshold_half_num,
        threshold_half_den,
        h->batch,
        d_rational_overflow_count);
    CUDA_CHECK_LAST_BOOL();
    CUDA_CHECK_SYNC_BOOL();
    return true;
}

static FStatHandle* fstat_create(
    int detector_rows_per_block,
    int batch,
    const InputType* d_in,
    float* d_out)
{
    if (detector_rows_per_block <= 0) {
        record_api_error("detector_rows_per_block must be positive.");
        return nullptr;
    }
    if (batch < 1) {
        record_api_error("batch must be at least one.");
        return nullptr;
    }
    if (d_in == nullptr) {
        record_api_error("input pointer is null.");
        return nullptr;
    }

    int device = 0;
    if (!record_cuda_error(cudaGetDevice(&device), __FILE__, __LINE__)) {
        return nullptr;
    }
    cudaDeviceProp prop;
    if (!record_cuda_error(cudaGetDeviceProperties(&prop, device), __FILE__, __LINE__)) {
        return nullptr;
    }
    if (batch > prop.maxGridSize[1]) {
        record_api_error("batch exceeds CUDA grid.y limit.");
        return nullptr;
    }

    FStatHandle* h = new (std::nothrow) FStatHandle;
    if (!h) {
        record_api_error("host allocation failed.");
        return nullptr;
    }

    h->detector_rows_per_block = detector_rows_per_block;
    h->batch     = batch;
    h->d_in      = d_in;
    h->d_out     = d_out;
    h->d_power_scratch = nullptr;
    h->d_weights = nullptr;
    h->d_weight_lanes = nullptr;
    h->weights_cached = false;

    if (!record_cuda_error(cudaMalloc(
        &h->d_power_scratch,
        batch * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long)), __FILE__,
    __LINE__)) {
        delete h;
        return nullptr;
    }
    if (!record_cuda_error(cudaMemset(
        h->d_power_scratch,
        0,
        batch * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long)), __FILE__,
    __LINE__)) {
        cudaFree(h->d_power_scratch);
        delete h;
        return nullptr;
    }
    #if FSTAT_USE_DP4A && !FSTAT_USE_CONSTANT_WEIGHT_LANES
    if (!record_cuda_error(cudaMalloc(
            &h->d_weight_lanes,
            FSTAT_WEIGHT_LANE_COUNT * sizeof(int)),
            __FILE__,
            __LINE__)) {
        cudaFree(h->d_power_scratch);
        delete h;
        return nullptr;
    }
    #elif FSTAT_USE_DP4A && FSTAT_USE_CONSTANT_WEIGHT_LANES
    // No per-handle weight storage is required. Weight lanes are copied into
    // module-global constant memory before each launch.
    #else
    if (!record_cuda_error(cudaMalloc(
            &h->d_weights,
            FSTAT_WEIGHT_BYTES),
            __FILE__,
            __LINE__)) {
        cudaFree(h->d_power_scratch);
        delete h;
        return nullptr;
    }
    #endif

    return h;
}

/* ===========================================================================
 * C API IMPLEMENTATION
 * ===========================================================================*/

extern "C" {

void FStat_GetSpecs(
    int* detector_window_samples,
    int* num_weight_terms,
    int* sample_bits_per_component,
    int* reference_offset_bins)
{
    clear_last_error();
    if (detector_window_samples) {
        *detector_window_samples = FSTAT_DETECTOR_WINDOW_SAMPLES;
    }
    if (num_weight_terms) {
        *num_weight_terms = FSTAT_NUM_WEIGHT_TERMS;
    }
    if (sample_bits_per_component) {
        *sample_bits_per_component = FSTAT_SAMPLE_BITS_PER_COMPONENT;
    }
    if (reference_offset_bins) {
        *reference_offset_bins = FSTAT_REFERENCE_BIN_OFFSET;
    }
}

void FStat_GetFeatures(
    int* use_dp4a,
    int* use_uint64_power_accumulation,
    int* block_threads)
{
    clear_last_error();
    if (use_dp4a) {
        *use_dp4a = FSTAT_USE_DP4A;
    }
    if (use_uint64_power_accumulation) {
        *use_uint64_power_accumulation = FSTAT_USE_UINT64_POWER_ACCUMULATION;
    }
    if (block_threads) {
        *block_threads = FSTAT_BLOCK_THREADS;
    }
}

void FStat_GetOptimizationFeatures(
    int* use_constant_weight_lanes,
    int* use_shared_weight_lanes,
    int* grid_max_blocks)
{
    clear_last_error();
    if (use_constant_weight_lanes) {
        *use_constant_weight_lanes = FSTAT_USE_CONSTANT_WEIGHT_LANES;
    }
    if (use_shared_weight_lanes) {
        *use_shared_weight_lanes = FSTAT_USE_SHARED_WEIGHT_LANES;
    }
    if (grid_max_blocks) {
        *grid_max_blocks = FSTAT_GRID_MAX_BLOCKS;
    }
}

void FStat_GetVersion(int* major, int* minor, int* patch)
{
    clear_last_error();
    if (major) {
        *major = FSTAT_CORE_VERSION_MAJOR;
    }
    if (minor) {
        *minor = FSTAT_CORE_VERSION_MINOR;
    }
    if (patch) {
        *patch = FSTAT_CORE_VERSION_PATCH;
    }
}

void* FStat_Create(
    const InputType* d_in,
    float* d_out,
    int detector_rows_per_block)
{
    clear_last_error();
    return static_cast<void*>(fstat_create(detector_rows_per_block, 1, d_in, d_out));
}

const char* FStat_LastError(void)
{
    return g_last_error;
}

void* FStat_Create_Batch(
    const InputType* d_in,
    float* d_out,
    int detector_rows_per_block,
    int batch)
{
    clear_last_error();
    return static_cast<void*>(fstat_create(detector_rows_per_block, batch, d_in, d_out));
}

void FStat_Destroy(void* handle)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) return;

    bool ok = true;

    if (h->d_power_scratch) {
        ok &= record_cuda_error(cudaFree(h->d_power_scratch), __FILE__, __LINE__);
        h->d_power_scratch = nullptr;
    }
    if (h->d_weights) {
        ok &= record_cuda_error(cudaFree(h->d_weights), __FILE__, __LINE__);
        h->d_weights = nullptr;
    }
    if (h->d_weight_lanes) {
        ok &= record_cuda_error(cudaFree(h->d_weight_lanes), __FILE__, __LINE__);
        h->d_weight_lanes = nullptr;
    }
    delete h;
    (void)ok;
}

void FStat_Compute_DiagnosticFloat(void* handle, const InputType* w_in)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;
    if (!fstat_write_f_statistic(h)) return;
}

void FStat_Compute_NumDen_Mask_RationalHalf(
    void* handle,
    const InputType* w_in,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    unsigned long long* d_num_out,
    unsigned long long* d_den_out,
    unsigned char* d_mask_out)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (d_num_out == nullptr) {
        record_api_error("numerator output pointer is null.");
        return;
    }
    if (d_den_out == nullptr) {
        record_api_error("denominator output pointer is null.");
        return;
    }
    if (d_mask_out == nullptr) {
        record_api_error("mask output pointer is null.");
        return;
    }
    if (threshold_half_den == 0ULL) {
        record_api_error("threshold_half_den must be nonzero.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;
    if (!fstat_write_numden_mask_rational(
        h,
        threshold_half_num,
        threshold_half_den,
        d_num_out,
        d_den_out,
        d_mask_out,
        nullptr)) return;
}

void FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount(
    void* handle,
    const InputType* w_in,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    unsigned long long* d_num_out,
    unsigned long long* d_den_out,
    unsigned char* d_mask_out,
    unsigned int* d_rational_overflow_count)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (d_num_out == nullptr) {
        record_api_error("numerator output pointer is null.");
        return;
    }
    if (d_den_out == nullptr) {
        record_api_error("denominator output pointer is null.");
        return;
    }
    if (d_mask_out == nullptr) {
        record_api_error("mask output pointer is null.");
        return;
    }
    if (d_rational_overflow_count == nullptr) {
        record_api_error("rational overflow-count pointer is null.");
        return;
    }
    if (threshold_half_den == 0ULL) {
        record_api_error("threshold_half_den must be nonzero.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;
    if (!fstat_write_numden_mask_rational(
        h,
        threshold_half_num,
        threshold_half_den,
        d_num_out,
        d_den_out,
        d_mask_out,
        d_rational_overflow_count)) return;
}

void FStat_Compute_Powers(void* handle, const InputType* w_in)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (h->d_out == nullptr) {
        record_api_error("output pointer is null for float power output.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;

    const int count = h->batch * FSTAT_NUM_WEIGHT_TERMS;
    const int threads = FSTAT_OUTPUT_KERNEL_THREADS;
    const int blocks = (count + threads - 1) / threads;
    kernel_convert_power_terms_to_float<<<blocks, threads>>>(
        h->d_power_scratch,
        h->d_out,
        count);
    CUDA_CHECK_LAST();
    CUDA_CHECK_SYNC();
}

void FStat_Compute_Powers_U64(
    void* handle,
    const InputType* w_in,
    unsigned long long* d_power_out)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (d_power_out == nullptr) {
        record_api_error("uint64 power output pointer is null.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;

    CUDA_CHECK(cudaMemcpy(
        d_power_out,
        h->d_power_scratch,
        h->batch * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long),
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK_SYNC();
}

} // extern "C"
