from fractions import Fraction
import math

import numpy as np
import pytest
from scipy.stats import binom

from pilot_proxy.testbench.fine_validation_stats import (
    ALWAYS_MASKED_Q16,
    MAX_Q16,
    binomial_interval,
    empirical_null_threshold,
    exact_q16_null_threshold,
    exact_rank_frontier,
    false_alarm_validation,
    invert_frontier,
    paired_crossing_loss_bootstrap,
    raw_crossing_brackets,
)


@pytest.mark.parametrize('n', [1, 5, 100, 5000])
def test_exact_binomial_zero_all_closed_forms(n):
    empty = binomial_interval(0, n, side='upper')
    full = binomial_interval(n, n, side='lower')
    assert empty['lower'] == 0
    assert empty['upper'] == pytest.approx(-math.expm1(math.log(.05)/n))
    assert full['upper'] == 1
    assert full['lower'] == pytest.approx(.05**(1/n))
    assert binomial_interval(0, n)['upper'] == pytest.approx(1-.025**(1/n))


@pytest.mark.parametrize('k,n', [(1, 8), (4, 17), (97, 100), (50, 5000)])
def test_binomial_endpoints_invert_binomial_tails(k, n):
    ci = binomial_interval(k, n)
    assert binom.sf(k-1, n, ci['lower']) == pytest.approx(.025, rel=1e-9)
    assert binom.cdf(k, n, ci['upper']) == pytest.approx(.025, rel=1e-9)
    assert ci['lower'] < k/n < ci['upper']


@pytest.mark.parametrize('args', [(True, 2), (1, 0), (-1, 8), (9, 8), (1.0, 2)])
def test_binomial_rejects_invalid_counts(args):
    with pytest.raises((ValueError, TypeError)):
        binomial_interval(*args)


def test_family_cap_is_separate_from_nominal_and_pointwise_precision():
    single = false_alarm_validation(50, 5000, nominal_pfa=.01, cap_pfa=.02)
    family = false_alarm_validation(50, 5000, nominal_pfa=.01, cap_pfa=.02, family_tests=184)
    assert family['simultaneous_upper']['upper'] > single['simultaneous_upper']['upper']
    assert family['cap_demonstrated']
    assert family['nominal_pfa'] == .01 and family['cap_pfa'] == .02
    assert family['pointwise'] == single['pointwise']
    assert not family['physical_acceptance']
    assert not false_alarm_validation(200, 5000, nominal_pfa=.01, cap_pfa=.02)['cap_demonstrated']


@pytest.mark.parametrize('n,p', [(10, Fraction(1, 10)), (2000, Fraction(1, 100)), (2000, Fraction(1, 20)), (5, Fraction(1, 3))])
def test_float_higher_threshold_matches_observed_order_and_keeps_ties(n, p):
    values = np.arange(n, dtype=float) / 10
    before = values.copy()
    result = empirical_null_threshold(values, pfa=p)
    exact_index = ((1-p).numerator*(n-1)+(1-p).denominator-1)//(1-p).denominator
    assert result['index'] == exact_index
    assert result['threshold'] == values[exact_index]
    assert result['exceedances'] == n-exact_index-1
    np.testing.assert_array_equal(values, before)
    repeated = empirical_null_threshold([1., 1., 1., 2., 2.], pfa=.4)
    assert repeated['threshold'] == 2 and repeated['exceedances'] == 0


@pytest.mark.parametrize('bad', [[1, np.inf], [1, np.nan], [], [-1, 1], [True, 1], [1+2j]])
def test_null_calibration_never_drops_or_coerces_invalids(bad):
    with pytest.raises((ValueError, TypeError)):
        empirical_null_threshold(bad, pfa=.01)


def test_q16_threshold_preserves_adjacent_uint64_maxima_and_sentinel():
    values = np.asarray([MAX_Q16-2, MAX_Q16-1, MAX_Q16], dtype=np.uint64)
    selected = exact_q16_null_threshold(values, pfa=Fraction(1, 2))
    assert selected['threshold'] == MAX_Q16-1
    assert selected['exceedances'] == 1
    unavailable = exact_q16_null_threshold([1, 2, ALWAYS_MASKED_Q16], pfa=.1)
    assert unavailable['status'] == 'unavailable'
    assert unavailable['threshold'] is None
    assert unavailable['selected_requirement'] == ALWAYS_MASKED_Q16
    assert unavailable['trials'] == 3 and unavailable['sentinel_trials'] == 1
    kept_sentinel = exact_q16_null_threshold([1, 2, 3, ALWAYS_MASKED_Q16], pfa=.5)
    assert kept_sentinel['threshold'] == 3 and kept_sentinel['exceedances'] == 1


@pytest.mark.parametrize('bad', [[0], [False], [2.0], [ALWAYS_MASKED_Q16+1], []])
def test_q16_refuses_undecoded_zero_or_noninteger_requirement(bad):
    with pytest.raises((TypeError, ValueError)):
        exact_q16_null_threshold(bad, pfa=.01)


def test_raw_crossing_preserves_downward_fluctuation_without_cummax():
    result = raw_crossing_brackets([0, 1, 2, 3], [.1, .4, .2, .8], target=.5)
    assert result['status'] == 'unique'
    assert result['upward_brackets'][0]['indices'] == [2, 3]
    assert result['estimate_db'] == pytest.approx(2.5)
    assert result['raw_rates'] == [.1, .4, .2, .8]


@pytest.mark.parametrize('rates', [[.1, .9, .1, .9], [.9, .1, .9, 1], [.1, .5, .5, .9], [.1, .5, .1, .9]])
def test_multiple_crossings_plateau_or_touch_are_ambiguous(rates):
    result = raw_crossing_brackets(range(len(rates)), rates, target=.5)
    assert result['status'] == 'ambiguous'
    assert result['estimate_db'] is None


@pytest.mark.parametrize('rates', [[.1, .2], [.8, .9], [.5, .9]])
def test_unbracketed_crossing_has_no_extrapolated_estimate(rates):
    result = raw_crossing_brackets([0, 1], rates, target=.5)
    assert result['status'] == 'unbracketed'
    assert result['estimate_db'] is None


def test_exact_target_single_hit_is_one_bracket():
    result = raw_crossing_brackets([0, 1, 2], [.1, .5, .9], target=.5)
    assert result['status'] == 'unique' and result['estimate_db'] == 1
    assert result['exact_target_indices'] == [1]


@pytest.mark.parametrize('x,y', [([1, 0], [.1, .9]), ([0, 0], [.1, .9]), ([0, 1], [.1, np.nan]), ([0, 1], [-.1, .9]), ([0], [.5])])
def test_crossing_invalid_grid_refuses(x, y):
    with pytest.raises(ValueError):
        raw_crossing_brackets(x, y, target=.5)


def paired_fixture():
    n = 20
    return dict(snr_db=[0., 1., 2.], null_by_stage={'a': np.ones(n), 'b': np.ones(n)},
                h1_by_stage={'a': np.tile([[0.], [2.], [2.]], (1, n)),
                             'b': np.tile([[0.], [0.], [2.]], (1, n))},
                trial_ids=[f'h{i}' for i in range(n)], null_ids=[f'n{i}' for i in range(n)],
                pfa=.05, target=.5, replicates=100, seed=845)


def test_paired_known_translation_has_exact_one_db_loss():
    args = paired_fixture()
    result = paired_crossing_loss_bootstrap(**args)
    pair = result['comparisons'][0]
    assert pair['valid_replicates'] == 100 and pair['censored_replicates'] == 0
    assert pair['replicate_losses_db'] == [1.]*100
    assert pair['loss_lo_db'] == pair['loss_hi_db'] == pair['point_loss_db'] == 1
    assert paired_crossing_loss_bootstrap(**args) == result


def test_paired_identical_random_stages_keep_zero_loss_in_every_valid_replicate():
    args = paired_fixture()
    rng = np.random.default_rng(11)
    null = rng.uniform(.5, 1, 20)
    responses = np.vstack([np.zeros(20), rng.uniform(.7, 1.3, 20), np.full(20, 2)])
    args['null_by_stage'] = {'a': null, 'b': null.copy()}
    args['h1_by_stage'] = {'a': responses, 'b': responses.copy()}
    pair = paired_crossing_loss_bootstrap(**args)['comparisons'][0]
    assert pair['interval_reportable']
    assert set(pair['replicate_losses_db']) == {0.0}


def test_paired_q16_calibration_and_replay_never_converts_to_float():
    args = paired_fixture()
    n = 20
    args['stage_kinds'] = {'a': 'q16', 'b': 'q16'}
    args['null_by_stage'] = {'a': [MAX_Q16-1]*n, 'b': [MAX_Q16-1]*n}
    args['h1_by_stage'] = {'a': [[MAX_Q16-2]*n, [MAX_Q16]*n, [MAX_Q16]*n],
                           'b': [[MAX_Q16-2]*n, [MAX_Q16-2]*n, [ALWAYS_MASKED_Q16]*n]}
    pair = paired_crossing_loss_bootstrap(**args)['comparisons'][0]
    assert pair['replicate_losses_db'] == [1.]*100


def test_bootstrap_preserves_all_unbracketed_replicates_and_withholds_interval():
    args = paired_fixture()
    args['h1_by_stage']['b'][:] = 0
    pair = paired_crossing_loss_bootstrap(**args)['comparisons'][0]
    assert pair['censored_replicates'] == 100
    assert pair['replicate_losses_db'] == [None]*100
    assert not pair['interval_reportable']
    assert pair['loss_lo_db'] is pair['loss_hi_db'] is pair['point_loss_db'] is None
    assert sum(pair['censoring_reasons'].values()) == 100


def test_bootstrap_censoring_fraction_cannot_be_hidden_by_valid_only_percentiles():
    args = paired_fixture()
    args['h1_by_stage']['b'] = np.vstack([np.zeros(20), np.zeros(20), np.r_[np.ones(10)*2, np.zeros(10)]])
    args['replicates'] = 400
    pair = paired_crossing_loss_bootstrap(**args)['comparisons'][0]
    assert 0 < pair['valid_replicates'] < 380
    assert len(pair['replicate_losses_db']) == 400
    assert pair['valid_replicates'] + pair['censored_replicates'] == 400
    assert not pair['interval_reportable'] and pair['loss_lo_db'] is None


def test_bootstrap_unavailable_threshold_is_censored_not_dropped():
    args = paired_fixture()
    args['stage_kinds'] = {'a': 'float', 'b': 'q16'}
    args['null_by_stage']['b'] = [ALWAYS_MASKED_Q16]*20
    args['h1_by_stage']['b'] = [[1]*20]*3
    pair = paired_crossing_loss_bootstrap(**args)['comparisons'][0]
    assert pair['valid_replicates'] == 0
    assert any('threshold_unavailable' in reason for reason in pair['censoring_reasons'])


@pytest.mark.parametrize('change', ['duplicate_null', 'overlap', 'short_h1', 'bad_fraction', 'bad_stage', 'duplicate_pair'])
def test_bootstrap_refuses_broken_pairing_and_weakened_censor_gate(change):
    args = paired_fixture()
    if change == 'duplicate_null': args['null_ids'][-1] = args['null_ids'][0]
    if change == 'overlap': args['null_ids'][-1] = args['trial_ids'][0]
    if change == 'short_h1': args['h1_by_stage']['b'] = args['h1_by_stage']['b'][:, :-1]
    if change == 'bad_fraction': args['min_valid_fraction'] = .5
    if change == 'bad_stage': args['stage_kinds'] = {'a': 'float', 'b': 'unknown'}
    if change == 'duplicate_pair': args['pairs'] = [('a', 'b'), ('a', 'b')]
    with pytest.raises((ValueError, TypeError)):
        paired_crossing_loss_bootstrap(**args)


def test_frontier_exact_steps_ties_sentinel_and_nonmonotone_mean():
    result = exact_rank_frontier({1: [2, 2, 7, ALWAYS_MASKED_Q16]}, [10., 0., 0., 100.], min_kept=1)
    rows = result['rows']
    assert [r['eta_q16'] for r in rows] == [1, 2, 7]
    assert [r['kept'] for r in rows] == [0, 2, 3]
    assert rows[0]['retained_truth_mean'] is None and not rows[0]['supported']
    assert rows[1]['retained_truth_mean'] == 5
    assert rows[2]['retained_truth_mean'] == pytest.approx(10/3)
    assert rows[2]['mask_only_cost'] == pytest.approx(4/3)
    assert result['sentinel_counts'] == {1: 1}


def test_frontier_exposure_and_total_cost_are_same_kept_population():
    result = exact_rank_frontier({1: [1, 2, 3]}, [2., 4., 100.], exposure=[1., 3., 6.], variance=[0., 2., 20.], min_kept=1)
    middle = result['rows'][1]
    assert middle['retained_truth_mean_unweighted'] == 3
    assert middle['retained_truth_mean'] == 3.5
    assert middle['exposure_retention'] == .4
    assert middle['mask_only_cost'] == 2.5
    assert middle['variance_mean'] == 1.5
    assert middle['total_cost'] == 6.25


def test_frontier_keeps_adjacent_uint64_thresholds_exact():
    result = exact_rank_frontier({1: [MAX_Q16-1, MAX_Q16]}, [1, 2], min_kept=1)
    assert [r['eta_q16'] for r in result['rows']] == [1, MAX_Q16-1, MAX_Q16]
    assert [r['kept'] for r in result['rows']] == [0, 1, 2]


def test_inversion_searches_all_unsmoothed_nonmonotone_steps_and_ranks():
    result = exact_rank_frontier({1: [1, 2, 3], 2: [3, 2, 1]}, [9., 0., 0.], min_kept=1)
    inversion = invert_frontier(result, [0., 3., 20.])['tolerances']
    assert inversion[0]['per_rank'][1] is None
    assert inversion[0]['per_rank'][2]['kept'] == 2
    assert inversion[1]['per_rank'][1]['kept'] == 3
    assert inversion[1]['envelope']['kept'] == 3
    assert inversion[2]['envelope']['kept'] == 3


def test_total_cost_objective_can_differ_from_minimum_mask():
    result = exact_rank_frontier({1: [1, 2]}, [0., 0.], variance=[0., 20.], min_kept=1)
    pure = invert_frontier(result, [1.])['tolerances'][0]['envelope']
    total = invert_frontier(result, [1.], objective='total_cost')['tolerances'][0]['envelope']
    assert pure['kept'] == 2 and total['kept'] == 1
    no_var = exact_rank_frontier({1: [1, 2]}, [0., 0.], min_kept=1)
    with pytest.raises(ValueError, match='variance'):
        invert_frontier(no_var, [1.], objective='total_cost')


def test_support_failure_is_retained_but_excluded_from_inversion():
    result = exact_rank_frontier({1: [1, 2]}, [0., 0.], min_kept=3)
    assert len(result['rows']) == 2 and not any(r['supported'] for r in result['rows'])
    assert invert_frontier(result, [0.])['tolerances'][0]['envelope'] is None


@pytest.mark.parametrize('kwargs', [dict(truth=[np.nan, 1]), dict(truth=[1, 2], exposure=[0, 1]), dict(truth=[1]), dict(truth=[1, 2], variance=[-1, 0])])
def test_frontier_refuses_invalid_or_misaligned_population(kwargs):
    with pytest.raises((ValueError, TypeError)):
        exact_rank_frontier({1: [1, 2]}, **kwargs)
