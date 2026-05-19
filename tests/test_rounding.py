"""Tests for apply_rounding."""

from naive_ieee754.rounding import RoundingMode, apply_rounding


def test_no_truncation_needed():
    bits = [1, 0, 1]
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TO_NEAREST_EVEN, 0)
    assert result == [1, 0, 1]
    assert not lost


def test_truncate_zeros():
    bits = [1, 0, 1, 0, 0]
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TO_NEAREST_EVEN, 0)
    assert result == [1, 0, 1]
    assert not lost  # discarded bits are all 0


def test_rne_rounds_up_when_guard_and_sticky():
    # guard=1, sticky=1 → round up
    bits = [1, 0, 0, 1, 1]  # keep 3: [1,0,0], guard=1, sticky includes 1
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TO_NEAREST_EVEN, 0)
    assert result == [1, 0, 1]
    assert lost


def test_rne_tie_rounds_to_even_up():
    # Tie: guard=1, sticky=0, last kept bit = 1 → round up (make it even... wait, 1 is odd, so round up)
    bits = [1, 0, 1, 1, 0]  # keep 3: [1,0,1], guard=1, sticky=0, last_kept=1 → round up
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TO_NEAREST_EVEN, 0)
    assert result == [1, 1, 0]
    assert lost


def test_rne_tie_rounds_to_even_down():
    # Tie: guard=1, sticky=0, last kept bit = 0 → truncate (already even)
    bits = [1, 0, 0, 1, 0]  # keep 3: [1,0,0], guard=1, sticky=0, last_kept=0 → truncate
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TO_NEAREST_EVEN, 0)
    assert result == [1, 0, 0]
    assert lost


def test_round_toward_zero_always_truncates():
    bits = [1, 1, 1, 1, 1]
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TOWARD_ZERO, 0)
    assert result == [1, 1, 1]
    assert lost


def test_round_toward_pos_inf_positive():
    bits = [1, 0, 0, 1, 0]  # positive, guard=1 → round up
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TOWARD_POS_INF, 0)
    assert result == [1, 0, 1]
    assert lost


def test_round_toward_pos_inf_negative():
    # Negative number: round toward +inf means truncate (reduce magnitude)
    bits = [1, 0, 0, 1, 0]
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TOWARD_POS_INF, 1)
    assert result == [1, 0, 0]
    assert lost


def test_round_toward_neg_inf_negative():
    # Negative: round toward -inf means round up in magnitude
    bits = [1, 0, 0, 1, 0]
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TOWARD_NEG_INF, 1)
    assert result == [1, 0, 1]
    assert lost


def test_carry_out():
    # Rounding [1,1,1] up should produce a carry → [1, 0, 0, 0] (length 4)
    bits = [1, 1, 1, 1, 0]  # keep 3: [1,1,1], guard=1, tie: last=1 → round up
    result, lost = apply_rounding(bits, 3, RoundingMode.ROUND_TO_NEAREST_EVEN, 0)
    assert result == [1, 0, 0, 0]  # carry produced an extra leading 1
    assert lost
