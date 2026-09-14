@page numerical_contract Numerical Behavior

# Numerical behavior and edge cases

The numerical tests cover representative boundaries and generated cases. They do
not establish that every dtype combination or floating-point input is safe or
accurate. Shape validation and dtype promotion are separate from the numerical
validity of each element operation.

## Element arithmetic

An element operator receives the original operand types. Native C++ integer
promotions and usual arithmetic conversions determine how it evaluates; the
result is then converted to the array's selected result dtype. A wider result
dtype does not retroactively widen the calculation. For example, adding an
`int32` value of `-3` to a `uint32` value of `1` evaluates in `uint32` and stores
`4294967294` in the promoted `int64` result. Adding `float32(16777216)` to
`int32(1)` evaluates in `float32`, so the promoted `float64` result is `16777216`.

Signed integer addition, subtraction, multiplication, unary negation, and the
corresponding reduction operations require representable native intermediate
results. These operations do not perform general overflow detection. Unsigned
expressions use native modular arithmetic; narrow integer operands can first
promote to signed `int`, so an unsigned storage dtype alone does not guarantee
that an intermediate calculation wraps safely. Converting floating-point values
to integers requires a finite, representable truncated value.

Division explicitly rejects zero divisors, including both signs of floating
zero and complex zero, with `Exceptions::ZeroDivisionError`. It also rejects
the minimum of a native signed expression type divided by minus one with
`Exceptions::OverflowError`, before evaluating that expression. This check uses
the native expression type: `int32` minimum divided by `int64(-1)` is valid,
while `int64` minimum divided by `int32(-1)` is rejected. A narrow signed minimum
divided by minus one can be valid in promoted `int` arithmetic; conversion back
to a narrow result dtype is a separate step.

Compound division validates every used operand pair before modifying elements.
Scalar zero divisors are rejected even for an empty left array. Array divisors
are checked only where the output consumes an element. These checks do not add
saturation or general floating-point overflow detection.

## Reductions

`sum` and `prod` use their `ReductionTraits` accumulator types, which default to
the input dtype. Callers needing a wider accumulator must select one explicitly
through a trait specialization or convert the input. They retain the native
overflow and rounding constraints of their actual accumulator expressions.

`mean` converts each input to `long double` before summing, divides in
`long double`, and returns `double`. Consequently, integer inputs never overflow
an integer sum accumulator while computing the mean. `ReductionTraits` does not
control mean accumulation. This is ordered, uncompensated accumulation: it can
still lose precision through cancellation or overflow in floating-point
arithmetic. Some platforms implement `long double` with the same precision and
range as `double`; results need not be bit-identical across platforms.

`var` computes population variance using Welford's algorithm with `double`
inputs and intermediates; `std` takes its square root. This avoids the direct
subtraction of two large squared means for ordinary large-offset data. Conversion
loss, rounding, and overflow remain possible, including when a large finite
input makes an intermediate difference or squared difference exceed `double`.
Neither operation promises an exact or compensated result. Axis reductions use
the same scalar algorithms for each slice.

On platforms supporting IEEE-style NaNs and infinities, statistics propagate
them through native arithmetic. A mean with a single infinity sign is infinite;
opposite infinities produce NaN. Variance of infinite input generally produces
NaN. These functions do not skip NaNs.

`min`, `max`, `argmin`, and `argmax` share an explicit scan in logical row-major
order. The first value initializes the candidate, and each later value replaces
it only when a strict comparison succeeds. With IEEE-style NaNs, an initial NaN
remains the selected
candidate; a later NaN does not replace a numeric candidate. This is neither a
NaN-skipping nor an unconditional NaN-propagating policy. Signed zeros compare
equal, so extrema retain the first zero's sign. The same scan handles owning
arrays, strided views, and each axis slice; it does not depend on a standard
library's vectorized extrema implementation. Sum and mean begin from positive
zero and do not promise to preserve the sign of an all-negative-zero input.

## Regression coverage

[The C++ cases](../../tests/cpp/stratax/ops/NumericEdges.cpp) exercise all eight
integer input dtypes for mean, every signed-eight-bit division pair whose divisor
is nonzero and whose quotient fits the result dtype, selected mixed-type
boundaries, checked division failures, strided reductions, cancellation,
nonfinite inputs, and signed zero. Extrema fixtures include 257-element inputs
for `float`, `double`, and `long double`, covering initial, later, and all NaNs,
repeated infinities, both initial zero signs, strided views, and axis slices.
The generated arithmetic cases keep native
intermediates representable, so sanitizer runs do not themselves invoke known
undefined operations. Expected integer values use a wider independent reference.

[The Python cases](../../tests/python/stratax/numeric_edges_tests.py) cover the
public `double` containers: large-offset statistics, nonfinite inputs, zero
signs, division failure atomicity, and exact power-of-two arithmetic. Python
integer constructor arguments are converted to `double`; they do not exercise
the C++ integer-container overflow checks. Python constructors and item assignment
reject nonfinite values with built-in `OverflowError`; the nonfinite reduction
cases produce infinities and NaNs through arithmetic on finite inputs.

Further work should choose an explicit policy before changing integer overflow
handling or pre-operation dtype conversion. An exhaustive floating-point
accuracy study would also need specified error bounds, reference arithmetic,
and platform coverage beyond this regression suite.
