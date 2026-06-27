/**
 * @file f_statistic.h
 * @brief Public C API for F-statistic CUDA kernel.
 *
 * Provides GPU-accelerated computation of the F-statistic for detecting
 * narrowband signals (e.g., DTV pilot tones) in wideband spectral data.
 *
 * @section usage Usage Example
 * @code
 *     // Query compile-time parameters
 *     int detector_window_samples, num_weight_terms, bits, reference_offset_bins;
 *     int core_major, core_minor, core_patch;
 *     FStat_GetSpecs(
 *         &detector_window_samples,
 *         &num_weight_terms,
 *         &bits,
 *         &reference_offset_bins);
 *     FStat_GetVersion(&core_major, &core_minor, &core_patch);
 *
 *     // Allocate GPU buffers for the detector-matrix view of one block
 *     InputType* d_input;  // row-major [detector_rows_per_block x detector_window_samples]
 *     float* d_output;     // Single F-statistic result
 *     cudaMalloc(
 *         &d_input,
 *         detector_rows_per_block * detector_window_samples * sizeof(InputType));
 *     cudaMalloc(&d_output, sizeof(float));
 *
 *     void* handle = FStat_Create(
 *         d_input, d_output, detector_rows_per_block);
 *     // For deployed NumDen/mask-only APIs, d_output may be NULL.
 *
 *     FStat_Compute_DiagnosticFloat(handle, host_weights);
 *     FStat_Destroy(handle);
 * @endcode
 *
 * @section dataformat Data Format
 * Input samples are packed complex integers:
 *   - High bits: Real component (signed)
 *   - Low bits:  Imaginary component (signed)
 *
 * For 4-bit mode (int8_t): bits [7:4] = real, bits [3:0] = imag.
 * Each nibble is interpreted as two's-complement signed int4 with range
 * [-8, 7]. If an upstream quantizer reserves or forbids the -8 code, that is
 * a data-generation constraint; the kernel decode itself implements the full
 * two's-complement int4 range.
 *
 * Weight ROM entries use the same packed complex format and are generated as
 * natural matched-filter weights. The CUDA kernel computes the conventional
 * complex dot product by multiplying each input sample by conj(weight).
 * Deployed 4-bit weight generation uses the symmetric component range [-7, 7]
 * even though the kernel decoder accepts [-8, 7], avoiding the negative-min
 * edge case in generated detector ROMs.
 *
 * Validated and frozen path: int8_t packed 4+4-bit complex samples.
 *
 * @section layout Detector Matrix Layout
 * All CUDA entry points consume a row-major detector matrix view.  For an
 * unbatched handle:
 *
 *   d_in[m, k] = d_in[m * detector_window_samples + k]
 *
 * where m is the detector row and k is the dot-product tap.  The 128-tap
 * detector window must be contiguous for each row.  For a batched handle:
 *
 *   d_in[b, m, k] =
 *       d_in[(b * detector_rows_per_block + m) * detector_window_samples + k]
 *
 * If an upstream ring buffer is naturally tap-major
 * [detector_window_samples x detector_rows_per_block], materialize or
 * transpose it into this row-major detector-matrix view before calling the
 * kernel.
 *
 * Host code that materializes a detector-matrix view from a contiguous
 * ring-buffer segment must ensure:
 *
 *   contiguous_sample_count % detector_window_samples == 0
 *   detector_rows_per_block = contiguous_sample_count / detector_window_samples
 *
 * If the ring buffer wraps, split the call at the physical wrap boundary or
 * stage into a contiguous buffer. A 128-sample detector row must not cross a
 * physical wrap boundary unless a future kernel explicitly adds modulo
 * addressing.
 *
 * @section refbins Reference Bin Contract
 * The locked detector uses two total reference bins: one lower reference bin
 * and one upper reference bin. With three weight terms, the raw F-statistic is
 * F = 2 * P_target / (P_ref1 + P_ref2).
 *
 * @section execution Execution Policy
 * Handles are device-affine and default-stream only. The caller must set the
 * intended CUDA device before FStat_Create/FStat_Create_Batch and keep the
 * same current device for compute and destroy calls on that handle. One host
 * thread must not use the same handle concurrently.
 *
 * @author Dylan
 * @date 2025
 */

#pragma once

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * QUERY FUNCTIONS
 * ===========================================================================*/

/**
 * @brief Query compile-time kernel parameters.
 *
 * Returns the values baked into the compiled library. Use this to ensure
 * host code matches kernel expectations.
 *
 * @param[out] detector_window_samples   Detector-window samples.
 * @param[out] num_weight_terms          Number of weight terms.
 * @param[out] sample_bits_per_component Bit depth per component.
 * @param[out] reference_offset_bins     Reference offset, in detector bins, used by the
 *                                       default target/reference artifact.
 *
 * @note Any parameter can be NULL if not needed.
 */
void FStat_GetSpecs(
    int* detector_window_samples,
    int* num_weight_terms,
    int* sample_bits_per_component,
    int* reference_offset_bins);

/**
 * @brief Query compiled kernel feature switches.
 *
 * Returns the implementation features baked into the compiled library. These
 * are separate from the scientific detector geometry returned by FStat_GetSpecs.
 *
 * @param[out] use_dp4a                       Nonzero when the DP4A dot-product path is compiled in.
 * @param[out] use_uint64_power_accumulation  Nonzero when integer power accumulation is compiled in.
 * @param[out] block_threads                  CUDA threads per block.
 *
 * @note Any parameter can be NULL if not needed.
 */
void FStat_GetFeatures(
    int* use_dp4a,
    int* use_uint64_power_accumulation,
    int* block_threads);

/**
 * @brief Query compiled launch/placement optimization switches.
 *
 * This is additive metadata for benchmarking and reporting. It does not change
 * the scientific detector geometry returned by FStat_GetSpecs.
 *
 * @param[out] use_constant_weight_lanes Nonzero when DP4A weight lanes are stored in CUDA constant memory.
 * @param[out] use_shared_weight_lanes   Nonzero when DP4A weight lanes are preloaded into shared memory per block.
 * @param[out] grid_max_blocks           Maximum grid blocks used by the power-accumulation launch.
 *
 * @note Any parameter can be NULL if not needed.
 */
void FStat_GetOptimizationFeatures(
    int* use_constant_weight_lanes,
    int* use_shared_weight_lanes,
    int* grid_max_blocks);

/**
 * @brief Query the locked core version.
 *
 * @note Any parameter can be NULL if not needed.
 */
void FStat_GetVersion(
    int* major,
    int* minor,
    int* patch);

/**
 * @brief Return the last CUDA/API error recorded by this thread.
 *
 * Public API entry points record CUDA/API failures here and return early.
 * Host wrappers can call this function to retrieve the error string instead
 * of having the process aborted. Public API entry points clear stale errors
 * before doing new work.
 */
const char* FStat_LastError(void);

typedef struct FStatRational {
    unsigned long long num;
    unsigned long long den;
} FStatRational;

/**
 * @brief Convert deployed NumDen output to the raw F-statistic.
 *
 * The deployed API writes num=P_target and den=P_ref1+P_ref2 for two total
 * reference bins: one lower reference bin and one upper reference bin. The
 * full raw F-statistic is therefore F = 2*num/den, not num/den. If den is
 * zero, this helper returns 0.0 to match the kernel's conservative
 * zero-reference policy.
 */
static inline double FStat_NumDenToRawF(
    unsigned long long num,
    unsigned long long den)
{
    if (den == 0ULL) {
        return 0.0;
    }
    return (FSTAT_RAW_NUMDEN_SCALE * (double)num) / (double)den;
}

/**
 * @brief Convert deployed NumDen output to linear pilot excess, F - 1.
 *
 * If den is zero, this helper returns 0.0 to match the kernel's conservative
 * zero-reference policy. Use FStat_NumDenToPilotExcessChecked when host code
 * needs to distinguish an invalid reference floor from valid zero excess.
 */
static inline double FStat_NumDenToPilotExcess(
    unsigned long long num,
    unsigned long long den)
{
    if (den == 0ULL) {
        return 0.0;
    }
    return FStat_NumDenToRawF(num, den) - FSTAT_NO_PILOT_EXCESS_RAW_F;
}

/**
 * @brief Checked conversion from deployed NumDen output to linear pilot excess.
 *
 * Returns 1 on success and writes `out`. Returns 0 for a null output pointer or
 * a zero denominator; on failure with a non-null output pointer, writes 0.0.
 */
static inline int FStat_NumDenToPilotExcessChecked(
    unsigned long long num,
    unsigned long long den,
    double* out)
{
    if (out == 0) {
        return 0;
    }
    if (den == 0ULL) {
        *out = 0.0;
        return 0;
    }

    *out = FStat_NumDenToRawF(num, den) - FSTAT_NO_PILOT_EXCESS_RAW_F;
    return 1;
}

/**
 * @brief Checked full-threshold to half-threshold conversion.
 *
 * Returns 1 on success and writes `out`. Returns 0 for a null output pointer,
 * zero denominator, or unsigned-long-long overflow in 2 * full_den. On failure
 * with a non-null output pointer, writes {0, 0}.
 */
static inline int FStat_MakeHalfThresholdFromFullChecked(
    unsigned long long full_num,
    unsigned long long full_den,
    FStatRational* out)
{
    if (out == 0) {
        return 0;
    }
    if (full_den == 0ULL) {
        out->num = 0ULL;
        out->den = 0ULL;
        return 0;
    }
    if (full_den > (~0ULL / 2ULL)) {
        out->num = 0ULL;
        out->den = 0ULL;
        return 0;
    }

    out->num = full_num;
    out->den = 2ULL * full_den;
    return 1;
}

/* ===========================================================================
 * LIFECYCLE FUNCTIONS
 * ===========================================================================*/

/**
 * @brief Create a handle for one detector-matrix view.
 *
 * `detector_rows_per_block` is the number of rows in the current contiguous
 * detector-matrix view for one detector block:
 *
 *   detector_rows_per_block = num_streams * samples_per_block /
 *                             detector_window_samples
 *
 * @param d_in  Device pointer to row-major input data
 *              [detector_rows_per_block x detector_window_samples], packed InputType
 * @param d_out Device pointer to output F-statistic (at least one float), or
 *              float-converted raw power terms (at least num_weight_terms
 *              floats) when using FStat_Compute_Powers. May be NULL when the
 *              handle is used exclusively with APIs that do not write
 *              floating-point diagnostic outputs, including:
 *                - FStat_Compute_NumDen_Mask_RationalHalf
 *                - FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount
 *                - FStat_Compute_Powers_U64
 * @param detector_rows_per_block Rows in the detector-matrix view of one block.
 *
 * @return Opaque handle pointer, or NULL on failure.
 *
 * @note d_in must remain valid for the lifetime of the handle. d_out must
 *       remain valid for APIs that write floating outputs.
 * @note The handle owns internal scratch memory which is freed by FStat_Destroy.
 * @note The handle is bound to the CUDA device current at creation time.
 */
void* FStat_Create(
    const InputType* d_in,
    float* d_out,
    int detector_rows_per_block);

/**
 * @brief Create a batched handle for detector-matrix views.
 *
 * Allocates internal GPU resources for computing F-statistics over multiple
 * independent input blocks in one launch. Each batch entry is treated as an
 * independent detector-matrix input.
 *
 * @param d_in   Device pointer to row-major input data
 *               [batch x detector_rows_per_block x detector_window_samples],
 *               packed InputType
 * @param d_out  Device pointer to output F-statistics [batch] (or
 *               [batch x num_weight_terms] when using FStat_Compute_Powers).
 *               May be NULL when the handle is used exclusively with APIs that
 *               do not write floating-point diagnostic outputs, including:
 *                 - FStat_Compute_NumDen_Mask_RationalHalf
 *                 - FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount
 *                 - FStat_Compute_Powers_U64
 * @param detector_rows_per_block Rows in each detector-matrix view.
 * @param batch  Number of independent blocks in the batch (>= 1)
 *
 * @return Opaque handle pointer, or NULL on failure.
 *
 * @note d_in must remain valid for the lifetime of the handle. d_out must
 *       remain valid for APIs that write floating outputs.
 * @note The handle owns internal scratch memory which is freed by FStat_Destroy.
 * @note The handle is bound to the CUDA device current at creation time.
 */
void* FStat_Create_Batch(
    const InputType* d_in,
    float* d_out,
    int detector_rows_per_block,
    int batch);

/**
 * @brief Destroy an F-statistic handle and free resources.
 *
 * @param handle Handle created by FStat_Create or FStat_Create_Batch
 *               (NULL is safely ignored).
 */
void FStat_Destroy(void* handle);

/* ===========================================================================
 * COMPUTATION FUNCTIONS
 * ===========================================================================*/

/**
 * @brief Compute diagnostic floating-point F-statistic for given weight vectors.
 *
 * Performs the following operations:
 *   1. Uploads/pre-packs weights for the compiled dot-product path
 *   2. Computes matched filter power for each weight vector in integer arithmetic
 *   3. Calculates F = 2 * P_target / (P_ref1 + P_ref2)
 *
 * The result is written to the d_out pointer specified at handle creation.
 * For batched handles, d_out must point to at least `batch` floats.
 * If P_ref1 + P_ref2 is zero, the diagnostic F-statistic output is 0.0f.
 *
 * @param handle Handle created by FStat_Create
 * @param w_in   Host pointer to packed weights [num_weight_terms x detector_window_samples] InputType
 *
 * @note This function is asynchronous. Call cudaDeviceSynchronize() if
 *       you need to read d_out immediately after.
 *
 * @note Weight layout:
 *       - first detector-window row  = target weights
 *       - second detector-window row = lower reference weights
 *       - third detector-window row  = upper reference weights
 */
void FStat_Compute_DiagnosticFloat(void* handle, const InputType* w_in);

/**
 * @brief Compute deployed fixed-point numerator, denominator, and mask.
 *
 * This is the production fixed-point detector product. It writes:
 *
 *   d_num_out[idx] = P_target
 *   d_den_out[idx] = P_ref1 + P_ref2
 *   d_mask_out[idx] =
 *       (P_target * threshold_half_den >=
 *        threshold_half_num * (P_ref1 + P_ref2)) ? 1 : 0
 *
 * The comparison is evaluated only when P_ref1 + P_ref2 is nonzero. If
 * d_den_out[idx] is zero, reference power is invalid for the deployed
 * detector and d_mask_out[idx] is forced to 0 regardless of P_target.
 * The raw F-statistic corresponding to these outputs is
 * FStat_NumDenToRawF(d_num_out[idx], d_den_out[idx]) = 2*num/den.
 *
 * This API takes threshold_half = F_threshold / 2. It does not take the full
 * F-statistic threshold. For example, a full raw F threshold of 1027/1024 is
 * passed as threshold_half_num=1027, threshold_half_den=2048.
 * threshold_half_den must be nonzero.
 *
 * The handle's d_out pointer is not used and may be NULL. For batched handles,
 * each output pointer must address `batch` elements.
 */
void FStat_Compute_NumDen_Mask_RationalHalf(
    void* handle,
    const InputType* w_in,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    unsigned long long* d_num_out,
    unsigned long long* d_den_out,
    unsigned char* d_mask_out);

/**
 * @brief Same as FStat_Compute_NumDen_Mask_RationalHalf, with overflow telemetry.
 *
 * d_rational_overflow_count must point to one device unsigned int. The API
 * clears it to zero at entry, then the rational comparison increments it for
 * each defensive uint64 saturation. Any nonzero value is a validation failure
 * for deployed thresholds.
 */
void FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount(
    void* handle,
    const InputType* w_in,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    unsigned long long* d_num_out,
    unsigned long long* d_den_out,
    unsigned char* d_mask_out,
    unsigned int* d_rational_overflow_count);

/**
 * @brief Compute raw power terms for each weight vector.
 *
 * Performs the same integer accumulation as the deployed path, then converts the
 * per-weight power terms to float in d_out (length num_weight_terms).
 * For batched handles, d_out must point to `batch * num_weight_terms` floats.
 *
 * @param handle Handle created by FStat_Create
 * @param w_in   Host pointer to packed weights [num_weight_terms x detector_window_samples] InputType
 *
 * @note d_out must point to at least num_weight_terms floats when using this function.
 */
void FStat_Compute_Powers(void* handle, const InputType* w_in);

/**
 * @brief Compute exact raw uint64 power terms for each weight vector.
 *
 * Performs the same integer accumulation as the deployed path, then copies the
 * internal raw power terms to d_power_out without converting them to float.
 * For batched handles, d_power_out must point to
 * `batch * num_weight_terms` unsigned long long values on the device.
 *
 * @param handle      Handle created by FStat_Create
 * @param w_in        Host pointer to packed weights [num_weight_terms x detector_window_samples] InputType
 * @param d_power_out Device pointer receiving raw uint64 power terms.
 *
 * @note This API is intended for exact diagnostics and regression tests. The
 *       deployed detector product is the NumDen half-threshold API.
 */
void FStat_Compute_Powers_U64(
    void* handle,
    const InputType* w_in,
    unsigned long long* d_power_out);

#ifdef __cplusplus
}
#endif
