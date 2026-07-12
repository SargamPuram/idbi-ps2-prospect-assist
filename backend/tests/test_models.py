"""
Boundary-value tests for backend/scoring/models.py.

Every threshold exercised here is read directly from the source, not guessed:
  - RAG_HOT_THRESHOLD (75) / RAG_WARM_THRESHOLD (50) -- scoring/constants.py,
    consumed by classify_rag_status().
  - The 0.40 "full disposable-income headroom" retained-ratio cutoff in
    calculate_capacity_score() -- hardcoded in scoring/models.py (there is no
    separate constants.py entry for it; the docstring there calls it out
    explicitly as "same headroom the previous FOIR-style disposable_ratio
    used").
  - The >=4 / >=2 sources-present High/Medium/Low cutoffs in
    calculate_confidence() -- hardcoded in scoring/models.py, deliberately
    mirroring PS3's confidence_level thresholds per that function's docstring.
  - months_of_data_available >= 4 and account_tenure_years >= 1 -- the two
    per-source thresholds inside calculate_confidence().

No network calls, no trained model, no CSV I/O -- every test builds a plain
dict "row" (as calculate_scores() would pass via df.apply(..., axis=1)) and
calls the pure per-row functions directly.
"""

import pytest

from scoring.constants import RAG_HOT_THRESHOLD, RAG_WARM_THRESHOLD, UPI_CATEGORIES
from scoring.models import (
    calculate_capacity_score, calculate_confidence, calculate_discipline_score, classify_rag_status,
)


def base_row(**overrides):
    """A fully-populated, all-zero customer row, so every function under test
    can safely index any upi_<category> / bureau / tenure field without a
    KeyError, then override just the fields a given test cares about."""
    row = {
        'employment_type': 'Salaried',
        'salary_credits': 0,
        'other_income_credits': 0,
        'total_emi_burden': 0,
        'pct_income_spent_within_3_days': 0.15,
        'days_to_balance_depletion': 20,
        'credit_bureau_available': 1,
        'months_of_data_available': 6,
        'account_tenure_years': 2,
    }
    row.update({f'upi_{c}': 0 for c in UPI_CATEGORIES})
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# calculate_capacity_score
# ---------------------------------------------------------------------------

class TestCapacityScoreIncomeBoundary:
    def test_zero_income_returns_zero(self):
        row = base_row(employment_type='Salaried', salary_credits=0)
        assert calculate_capacity_score(row) == 0

    def test_just_above_zero_income_is_nonzero(self):
        row = base_row(employment_type='Salaried', salary_credits=1)
        assert calculate_capacity_score(row) > 0

    def test_self_employed_uses_other_income_credits(self):
        # _get_income() branches on employment_type; Salaried with 0
        # salary_credits must not fall back to other_income_credits.
        row = base_row(employment_type='Salaried', salary_credits=0, other_income_credits=50000)
        assert calculate_capacity_score(row) == 0

        row = base_row(employment_type='Self-employed', salary_credits=0, other_income_credits=0)
        assert calculate_capacity_score(row) == 0


class TestCapacityScoreRetainedRatioBoundary:
    """
    base_score = (retained_ratio / 0.40) * 100, so retained_ratio == 0.40 is
    exactly where the disposable-income component of the score first hits
    its 100-point ceiling.
    """

    INCOME = 100_000

    def _row_at_retained_ratio(self, needs, wants, *, pct_3d=0.0, days_to_deplete=20):
        # Neutral (max, 100-point) discipline component by default, so the
        # capacity score isolates the retained-ratio behavior being tested.
        return base_row(
            employment_type='Salaried',
            salary_credits=self.INCOME,
            upi_groceries=needs,
            upi_shopping=wants,
            pct_income_spent_within_3_days=pct_3d,
            days_to_balance_depletion=days_to_deplete,
        )

    def test_exactly_at_040_retained_ratio_hits_full_100(self):
        # needs+wants = 60% of income -> retained_ratio == 0.40 exactly.
        row = self._row_at_retained_ratio(needs=40_000, wants=20_000)
        assert calculate_capacity_score(row) == pytest.approx(100.0)

    def test_just_below_040_retained_ratio_is_under_100(self):
        # 1000 rupees more spent on needs -> retained_ratio == 0.39, clearly
        # below 0.40 even after calculate_cash_flow_segments' 4-decimal
        # rounding of retained_income_ratio (a 1-rupee delta would round back
        # up to 0.4000 and defeat the point of this test).
        row = self._row_at_retained_ratio(needs=41_000, wants=20_000)
        score = calculate_capacity_score(row)
        assert score < 100.0
        assert score == pytest.approx(97.875, abs=1e-3)

    def test_zero_retained_ratio(self):
        # needs+wants == income exactly -> retained_amount == 0.
        row = self._row_at_retained_ratio(needs=60_000, wants=40_000)
        segments_score = calculate_capacity_score(row)
        # base_score == 0, discipline == 100 (neutral) -> 0.85*0 + 0.15*100 == 15
        assert segments_score == pytest.approx(15.0)

    def test_deep_deficit_retained_ratio_clips_to_zero_score(self):
        # needs+wants far exceeds income -> retained_income_ratio clipped at
        # -5 inside calculate_cash_flow_segments, driving a deeply negative
        # base_score; final np.clip(score, 0, 100) in calculate_capacity_score
        # must floor the result at 0, never go negative.
        row = self._row_at_retained_ratio(needs=500_000, wants=500_000)
        score = calculate_capacity_score(row)
        assert score == 0.0

    def test_score_never_exceeds_100_even_with_large_surplus(self):
        # wants=needs=0 -> retained_ratio == 1.0 (100% retained), far above
        # the 0.40 ceiling -> base_score would be 250 uncapped.
        row = self._row_at_retained_ratio(needs=0, wants=0)
        assert calculate_capacity_score(row) == pytest.approx(100.0)


class TestDisciplineScoreBoundary:
    """
    velocity_score = (1 - pct_3d) * 100          -- 0 at pct_3d=1.0, 100 at pct_3d=0.0
    depletion_score = clip(days_to_deplete/20*100, 0, 100)  -- saturates at days=20
    score = 0.6*velocity_score + 0.4*depletion_score
    """

    def test_best_case_pct_3d_zero_and_full_depletion_window(self):
        row = base_row(pct_income_spent_within_3_days=0.0, days_to_balance_depletion=20)
        assert calculate_discipline_score(row) == pytest.approx(100.0)

    def test_worst_case_pct_3d_one_and_immediate_depletion(self):
        row = base_row(pct_income_spent_within_3_days=1.0, days_to_balance_depletion=0)
        assert calculate_discipline_score(row) == pytest.approx(0.0)

    def test_depletion_score_saturates_beyond_20_days(self):
        # IDBI's own "salary spent immediately" red flag is about the fast
        # end of this scale, but the slow end must not reward beyond 20 days
        # either -- depletion_score is explicitly capped at 100.
        row_at_20 = base_row(pct_income_spent_within_3_days=0.0, days_to_balance_depletion=20)
        row_at_28 = base_row(pct_income_spent_within_3_days=0.0, days_to_balance_depletion=28)
        assert calculate_discipline_score(row_at_20) == calculate_discipline_score(row_at_28)

    def test_red_flag_pattern_scores_low(self):
        # IDBI's worked example: salary credited day one, most of it spent
        # within days. pct_3d=0.70 (> the 0.60 low_financial_discipline_flag
        # cutoff used in generate_data.py) with a 5-day depletion window.
        row = base_row(pct_income_spent_within_3_days=0.70, days_to_balance_depletion=5)
        score = calculate_discipline_score(row)
        assert score < 50.0


# ---------------------------------------------------------------------------
# calculate_confidence
# ---------------------------------------------------------------------------

class TestConfidenceLevelBoundary:
    """
    5 independent sources checked; level is High at >=4, Medium at >=2, else
    Low. These cutoffs are hardcoded in calculate_confidence() itself.
    """

    def _row_with_sources(self, n_sources):
        """Build a row with exactly n_sources (0-5) of the 5 confidence
        signals present, by toggling each independently."""
        has_income = n_sources >= 1
        has_upi = n_sources >= 2
        has_bureau = n_sources >= 3
        has_history = n_sources >= 4
        has_tenure = n_sources >= 5
        return base_row(
            employment_type='Salaried',
            salary_credits=50_000 if has_income else 0,
            upi_groceries=1000 if has_upi else 0,
            credit_bureau_available=1 if has_bureau else 0,
            months_of_data_available=6 if has_history else 1,
            account_tenure_years=3 if has_tenure else 0,
        )

    @pytest.mark.parametrize("n_sources,expected_level", [
        (5, 'High'),
        (4, 'High'),   # exact >=4 boundary
        (3, 'Medium'),  # just below the High boundary
        (2, 'Medium'),  # exact >=2 boundary
        (1, 'Low'),     # just below the Medium boundary
        (0, 'Low'),
    ])
    def test_level_boundaries(self, n_sources, expected_level):
        row = self._row_with_sources(n_sources)
        _, level = calculate_confidence(row)
        assert level == expected_level

    def test_data_completeness_score_is_percentage_of_5(self):
        row = self._row_with_sources(4)
        score, _ = calculate_confidence(row)
        assert score == pytest.approx(80.0)  # 4/5 * 100

    def test_months_of_data_available_exact_boundary(self):
        # >=4 months counts as "sufficient transaction history"; 3 does not.
        row_3 = self._row_with_sources(3)
        row_3['months_of_data_available'] = 3
        row_4 = self._row_with_sources(3)
        row_4['months_of_data_available'] = 4
        score_3, _ = calculate_confidence(row_3)
        score_4, _ = calculate_confidence(row_4)
        assert score_4 > score_3

    def test_account_tenure_exact_one_year_boundary(self):
        # >=1 year counts as "established relationship"; 0 does not.
        row_0 = self._row_with_sources(4)
        row_0['account_tenure_years'] = 0
        row_1 = self._row_with_sources(4)
        row_1['account_tenure_years'] = 1
        score_0, _ = calculate_confidence(row_0)
        score_1, _ = calculate_confidence(row_1)
        assert score_1 > score_0


# ---------------------------------------------------------------------------
# classify_rag_status (composite-score -> Hot/Warm/Cold)
# ---------------------------------------------------------------------------

class TestRagStatusBoundary:
    def test_exactly_at_hot_threshold_is_hot(self):
        assert classify_rag_status(RAG_HOT_THRESHOLD) == 'Hot'
        assert classify_rag_status(75.0) == 'Hot'

    def test_just_below_hot_threshold_is_warm(self):
        assert classify_rag_status(RAG_HOT_THRESHOLD - 0.1) == 'Warm'
        assert classify_rag_status(74.9) == 'Warm'

    def test_exactly_at_warm_threshold_is_warm(self):
        assert classify_rag_status(RAG_WARM_THRESHOLD) == 'Warm'
        assert classify_rag_status(50.0) == 'Warm'

    def test_just_below_warm_threshold_is_cold(self):
        assert classify_rag_status(RAG_WARM_THRESHOLD - 0.1) == 'Cold'
        assert classify_rag_status(49.9) == 'Cold'

    def test_extremes(self):
        assert classify_rag_status(0) == 'Cold'
        assert classify_rag_status(100) == 'Hot'
