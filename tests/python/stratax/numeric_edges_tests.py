import math
import operator

import pytest
import stratax as sx


def vector_with_nonfinite_results(values):
    # Constructors reject nonfinite inputs. Produce them through allowed
    # arithmetic on finite values: overflow gives infinity, inf-inf gives NaN.
    left = sx.Vector([value if math.isfinite(value) else 1.0e308 for value in values])
    multipliers = sx.Vector([
        1.0 if math.isfinite(value) else math.copysign(2.0, value)
        for value in values
    ])
    subtract = sx.Vector([1.0e308 if math.isnan(value) else 0.0 for value in values])
    return left * multipliers - subtract * 2.0


@pytest.mark.parametrize("offset", [0.0, 1.0e6, 1.0e12, -1.0e12])
def test_population_statistics_resist_large_offset_cancellation(offset):
    values = sx.Vector([offset + value for value in (1.0, 2.0, 3.0, 4.0)])
    assert sx.mean(values) == offset + 2.5
    assert sx.var(values) == 1.25
    assert sx.std(values) == math.sqrt(1.25)


def test_mean_and_axis_mean_retain_integer_boundary_values_in_float_storage():
    # Python's public numeric containers store double, including integer inputs.
    largest_int32 = 2**31 - 1
    values = sx.Matrix([[largest_int32, largest_int32], [-2**31, -2**31]])
    assert sx.mean(values) == -0.5
    assert sx.mean(values, 0).tolist() == [-0.5, -0.5]
    assert sx.mean(values, 1, True).tolist() == [[float(largest_int32)], [float(-2**31)]]
    assert sx.mean(sx.Vector([1.0e16, -1.0e16, 1.0])) == 1.0 / 3.0


@pytest.mark.parametrize("values,expected", [
    ([math.inf, 1.0], math.inf),
    ([-math.inf, 1.0], -math.inf),
])
def test_mean_preserves_single_infinity_sign(values, expected):
    assert sx.mean(vector_with_nonfinite_results(values)) == expected


@pytest.mark.parametrize("name", ["mean", "var", "std"])
@pytest.mark.parametrize("values", [[1.0, math.nan], [math.nan, 1.0], [math.inf, -math.inf]])
def test_nonfinite_statistics_are_not_nan_skipping(name, values):
    assert math.isnan(getattr(sx, name)(vector_with_nonfinite_results(values)))


@pytest.mark.parametrize("name", ["var", "std"])
def test_infinite_constant_population_has_nan_variance(name):
    values = vector_with_nonfinite_results([math.inf, math.inf])
    assert math.isnan(getattr(sx, name)(values))


def test_extrema_nan_selection_follows_first_candidate_and_ordinary_comparisons():
    first_nan = vector_with_nonfinite_results([math.nan, 2.0, 1.0])
    later_nan = vector_with_nonfinite_results([2.0, math.nan, 1.0])
    assert math.isnan(sx.min(first_nan))
    assert math.isnan(sx.max(first_nan))
    assert sx.argmin(first_nan) == 0
    assert sx.argmax(first_nan) == 0
    assert sx.min(later_nan) == 1.0
    assert sx.max(later_nan) == 2.0
    assert sx.argmin(later_nan) == 2
    assert sx.argmax(later_nan) == 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_python_constructor_and_assignment_inputs_are_rejected(value):
    with pytest.raises(OverflowError):
        sx.Vector([value])
    source = sx.Vector([1.0])
    with pytest.raises(OverflowError):
        source[0] = value
    assert source[0] == 1.0


def test_signed_zero_survives_division_and_first_extrema_ties():
    values = sx.Vector([-0.0, 0.0])
    quotient = values / 2.0
    assert math.copysign(1.0, quotient[0]) == -1.0
    assert math.copysign(1.0, quotient[1]) == 1.0
    assert math.copysign(1.0, sx.min(values)) == -1.0
    assert math.copysign(1.0, sx.max(values)) == -1.0
    assert sx.argmin(values) == sx.argmax(values) == 0


@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_both_zero_signs_reject_every_division_arrangement_without_partial_mutation(zero):
    source = sx.Vector([8.0, 16.0])
    divisors = sx.Vector([2.0, zero])
    for divide in (
        lambda: source / zero,
        lambda: source / divisors,
        lambda: 1.0 / divisors,
        lambda: operator.itruediv(source, zero),
        lambda: operator.itruediv(source, divisors),
    ):
        with pytest.raises(sx.ZeroDivisionError):
            divide()
        assert source.tolist() == [8.0, 16.0]


@pytest.mark.parametrize("exponent", [-500, -20, 0, 20, 500])
def test_generated_exact_power_of_two_arithmetic(exponent):
    scale = math.ldexp(1.0, exponent)
    source = sx.Vector([scale, 2.0 * scale, -4.0 * scale])
    assert (source * 2.0).tolist() == [2.0 * scale, 4.0 * scale, -8.0 * scale]
    assert (source / 2.0).tolist() == [0.5 * scale, scale, -2.0 * scale]
    assert (source + source).tolist() == [2.0 * scale, 4.0 * scale, -8.0 * scale]
    assert (source - source).tolist() == [0.0, 0.0, 0.0]
