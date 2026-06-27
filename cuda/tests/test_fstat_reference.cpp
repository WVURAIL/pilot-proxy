#include "f_statistic.h"
#include "f_statistic_reference.h"

#include <cassert>

namespace {

constexpr int INT4_COMPONENT_MASK = 0xF;
constexpr int INT4_REAL_SHIFT_BITS = 4;
constexpr int INT4_ZERO_CODE = 0x0;
constexpr int INT4_MAX_POSITIVE_CODE = 0x7;
constexpr int INT4_MIN_NEGATIVE_CODE = 0x8;
constexpr int INT4_NEGATIVE_ONE_CODE = 0xF;
constexpr int INT4_ZERO_VALUE = 0;
constexpr int INT4_MAX_POSITIVE_VALUE = 7;
constexpr int INT4_MIN_NEGATIVE_VALUE = -8;
constexpr int INT4_NEGATIVE_ONE_VALUE = -1;
constexpr int UNINITIALIZED_SENTINEL = 999;
constexpr int SINGLE_DETECTOR_ROW = 1;

constexpr unsigned long long ZERO_POWER = 0ULL;
constexpr unsigned long long UNIT_COMPLEX_POWER = 1ULL;
constexpr unsigned long long TWO_COMPONENT_COMPLEX_POWER = 4ULL;
constexpr unsigned long long CHECKED_RATIONAL_SENTINEL_NUM = 123ULL;
constexpr unsigned long long CHECKED_RATIONAL_SENTINEL_DEN = 456ULL;
constexpr unsigned long long UNIT_THRESHOLD_NUM = 1ULL;
constexpr unsigned long long UNIT_THRESHOLD_DEN = 1ULL;
constexpr unsigned long long FULL_THRESHOLD_EXAMPLE_NUM = 1027ULL;
constexpr unsigned long long FULL_THRESHOLD_EXAMPLE_DEN = 1024ULL;
constexpr unsigned long long HALF_THRESHOLD_EXAMPLE_DEN = 2048ULL;
constexpr unsigned long long INVALID_THRESHOLD_DEN = 0ULL;
constexpr unsigned long long OVERFLOW_THRESHOLD_DEN = ~0ULL;
constexpr unsigned long long RAW_F_EXAMPLE_NUM = 3ULL;
constexpr unsigned long long RAW_F_EXAMPLE_DEN = 2ULL;
constexpr double RAW_F_EXAMPLE_VALUE = 3.0;
constexpr double PILOT_EXCESS_EXAMPLE_VALUE = 2.0;
constexpr double PILOT_EXCESS_SENTINEL = 123.0;

InputType pack_i4(int real, int imag)
{
    return static_cast<InputType>(
        ((real & INT4_COMPONENT_MASK) << INT4_REAL_SHIFT_BITS)
        | (imag & INT4_COMPONENT_MASK));
}

void assert_unpack(InputType packed, int expected_real, int expected_imag)
{
    int real = 0;
    int imag = 0;
    fstat_ref_unpack_complex_i4(packed, &real, &imag);
    assert(real == expected_real);
    assert(imag == expected_imag);
}

void assert_mul(
    int xr,
    int xi,
    int wr,
    int wi,
    int expected_real,
    int expected_imag)
{
    int real = UNINITIALIZED_SENTINEL;
    int imag = UNINITIALIZED_SENTINEL;
    fstat_ref_complex_mul_conj(xr, xi, wr, wi, &real, &imag);
    assert(real == expected_real);
    assert(imag == expected_imag);
}

void assert_single_tap_power(
    InputType x0,
    InputType w0,
    unsigned long long expected_power)
{
    InputType x[FSTAT_DETECTOR_WINDOW_SAMPLES] = {};
    InputType w[FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES] = {};
    x[0] = x0;
    w[0] = w0;

    unsigned long long powers[FSTAT_REFERENCE_POWER_TERMS] = {};
    fstat_ref_powers_u64(x, w, SINGLE_DETECTOR_ROW, powers);
    assert(powers[FSTAT_TARGET_WEIGHT_INDEX] == expected_power);
    assert(powers[FSTAT_LOWER_REFERENCE_WEIGHT_INDEX] == ZERO_POWER);
    assert(powers[FSTAT_UPPER_REFERENCE_WEIGHT_INDEX] == ZERO_POWER);
}

}  // namespace

int main()
{
    assert(fstat_ref_sign_extend_i4(INT4_ZERO_CODE) == INT4_ZERO_VALUE);
    assert(
        fstat_ref_sign_extend_i4(INT4_MAX_POSITIVE_CODE)
        == INT4_MAX_POSITIVE_VALUE);
    assert(
        fstat_ref_sign_extend_i4(INT4_MIN_NEGATIVE_CODE)
        == INT4_MIN_NEGATIVE_VALUE);
    assert(
        fstat_ref_sign_extend_i4(INT4_NEGATIVE_ONE_CODE)
        == INT4_NEGATIVE_ONE_VALUE);

    assert_unpack(pack_i4(-8, 7), -8, 7);
    assert_unpack(pack_i4(1, -1), 1, -1);

    assert_mul(1, 1, 1, 1, 2, 0);
    assert_mul(1, 0, 0, 1, 0, -1);
    assert_mul(0, 1, 1, 0, 0, 1);
    assert_mul(1, -1, 1, 1, 0, -2);

    assert_single_tap_power(
        pack_i4(1, 1), pack_i4(1, 1), TWO_COMPONENT_COMPLEX_POWER);
    assert_single_tap_power(
        pack_i4(1, 0), pack_i4(0, 1), UNIT_COMPLEX_POWER);
    assert_single_tap_power(
        pack_i4(0, 1), pack_i4(1, 0), UNIT_COMPLEX_POWER);
    assert_single_tap_power(
        pack_i4(1, -1), pack_i4(1, 1), TWO_COMPONENT_COMPLEX_POWER);

    FStatRational checked = {
        CHECKED_RATIONAL_SENTINEL_NUM,
        CHECKED_RATIONAL_SENTINEL_DEN};
    assert(FStat_MakeHalfThresholdFromFullChecked(
        FULL_THRESHOLD_EXAMPLE_NUM,
        FULL_THRESHOLD_EXAMPLE_DEN,
        &checked) == 1);
    assert(checked.num == FULL_THRESHOLD_EXAMPLE_NUM);
    assert(checked.den == HALF_THRESHOLD_EXAMPLE_DEN);

    assert(FStat_MakeHalfThresholdFromFullChecked(
        UNIT_THRESHOLD_NUM,
        INVALID_THRESHOLD_DEN,
        &checked) == 0);
    assert(checked.num == ZERO_POWER);
    assert(checked.den == ZERO_POWER);

    assert(FStat_MakeHalfThresholdFromFullChecked(
        UNIT_THRESHOLD_NUM,
        OVERFLOW_THRESHOLD_DEN,
        &checked) == 0);
    assert(checked.num == ZERO_POWER);
    assert(checked.den == ZERO_POWER);

    assert(FStat_MakeHalfThresholdFromFullChecked(
        UNIT_THRESHOLD_NUM,
        UNIT_THRESHOLD_DEN,
        0) == 0);
    assert(FStat_NumDenToRawF(RAW_F_EXAMPLE_NUM, RAW_F_EXAMPLE_DEN)
        == RAW_F_EXAMPLE_VALUE);
    assert(FStat_NumDenToRawF(RAW_F_EXAMPLE_NUM, INVALID_THRESHOLD_DEN) == 0.0);
    assert(FStat_NumDenToPilotExcess(RAW_F_EXAMPLE_NUM, RAW_F_EXAMPLE_DEN)
        == PILOT_EXCESS_EXAMPLE_VALUE);
    assert(FStat_NumDenToPilotExcess(RAW_F_EXAMPLE_NUM, INVALID_THRESHOLD_DEN)
        == 0.0);

    double rho = PILOT_EXCESS_SENTINEL;
    assert(FStat_NumDenToPilotExcessChecked(
        RAW_F_EXAMPLE_NUM,
        RAW_F_EXAMPLE_DEN,
        &rho) == 1);
    assert(rho == PILOT_EXCESS_EXAMPLE_VALUE);
    assert(FStat_NumDenToPilotExcessChecked(
        RAW_F_EXAMPLE_NUM,
        INVALID_THRESHOLD_DEN,
        &rho) == 0);
    assert(rho == 0.0);
    assert(FStat_NumDenToPilotExcessChecked(
        RAW_F_EXAMPLE_NUM,
        RAW_F_EXAMPLE_DEN,
        0) == 0);

    return 0;
}
