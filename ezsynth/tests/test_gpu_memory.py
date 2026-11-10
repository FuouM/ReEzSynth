import pytest

from ezsynth.utils.gpu_memory import MIN_BUDGET_MB, compute_budget_bytes

MIB = 1 << 20
TOTAL = 12288 * MIB


def test_budget_scales_with_memory_free_now_not_total():
    # Another app (browser) holds 4 GiB: the budget follows what is actually free.
    free = 8192 * MIB
    budget = compute_budget_bytes(free, TOTAL, fraction=0.8, reserve_bytes=1024 * MIB)
    assert budget == int(free * 0.8)


def test_reserve_is_kept_free_even_when_fraction_would_take_more():
    free = 4096 * MIB
    budget = compute_budget_bytes(free, TOTAL, fraction=1.0, reserve_bytes=1024 * MIB)
    assert budget == free - 1024 * MIB


def test_memory_already_held_by_this_process_counts_as_available():
    budget = compute_budget_bytes(
        2048 * MIB, TOTAL, 2048 * MIB, fraction=1.0, reserve_bytes=1024 * MIB
    )
    assert budget == 3072 * MIB


def test_explicit_limit_wins_but_never_eats_the_reserve():
    free = 6144 * MIB
    assert (
        compute_budget_bytes(
            free, TOTAL, reserve_bytes=1024 * MIB, limit_bytes=2048 * MIB
        )
        == 2048 * MIB
    )
    assert (
        compute_budget_bytes(
            free, TOTAL, reserve_bytes=1024 * MIB, limit_bytes=10_000 * MIB
        )
        == free - 1024 * MIB
    )


def test_too_little_free_memory_fails_fast():
    free = 1024 * MIB + (MIN_BUDGET_MB - 1) * MIB
    with pytest.raises(RuntimeError, match="GPU memory available"):
        compute_budget_bytes(free, TOTAL, reserve_bytes=1024 * MIB, fraction=1.0)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        compute_budget_bytes(8192 * MIB, TOTAL, fraction=fraction)
