#include <gtest/gtest.h>

#include <stratax.h>
#include <stratax/ops/Numeric.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

using namespace stratax::container;
using namespace stratax::core;
using stratax::core::numeric_detail::require_valid_division;

template<typename T>
class IntegerMeanEdges : public ::testing::Test {};

using IntegerMeanTypes = ::testing::Types<
	std::int8_t, std::int16_t, std::int32_t, std::int64_t,
	std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE(IntegerMeanEdges, IntegerMeanTypes);

TYPED_TEST(IntegerMeanEdges, RepeatedMaximumDoesNotOverflowInputAccumulator)
{
	using T = TypeParam;
	constexpr T highest = std::numeric_limits<T>::max();
	const Vector<T> source{highest, highest, highest, highest};
	static_assert(std::same_as<decltype(reduction::mean(source)), double>);
	EXPECT_DOUBLE_EQ(reduction::mean(source), static_cast<double>(highest));

	const Matrix<T> matrix{{highest, highest}, {highest, highest}};
	const auto along_rows = reduction::mean(matrix, 1, true);
	static_assert(std::same_as<typename decltype(along_rows)::value_type, double>);
	EXPECT_EQ(along_rows.shape(), Shape({2, 1}));
	EXPECT_DOUBLE_EQ(along_rows[0], static_cast<double>(highest));
	EXPECT_DOUBLE_EQ(along_rows[1], static_cast<double>(highest));
}

TYPED_TEST(IntegerMeanEdges, ConstantAndSymmetricInputsPreserveMean)
{
	using T = TypeParam;
	for (int value = 0; value <= 100; value += 5)
	{
		const T center = static_cast<T>(value);
		EXPECT_DOUBLE_EQ(reduction::mean(Vector<T>{center, center, center}), value);
	}
	if constexpr (std::is_signed_v<T>)
	{
		constexpr T lowest = std::numeric_limits<T>::min();
		constexpr T highest = std::numeric_limits<T>::max();
		EXPECT_DOUBLE_EQ(reduction::mean(Vector<T>{lowest, lowest}),
			static_cast<double>(lowest));
		EXPECT_DOUBLE_EQ(reduction::mean(Vector<T>{highest, static_cast<T>(-highest)}), 0.0);
	}
}

TEST(NumericMeanEdges, IntegerExtremesAndStridedAxisSlices)
{
	constexpr auto lowest = std::numeric_limits<std::int32_t>::min();
	constexpr auto highest = std::numeric_limits<std::int32_t>::max();
	Matrix<std::int32_t> source{{lowest, highest, lowest}, {highest, lowest, highest}};
	const auto columns = reduction::mean(source, 0);
	for (const auto value : columns)
	{
		EXPECT_DOUBLE_EQ(value, -0.5);
	}
	auto view = stratax::indexing::slice(source, Slice{0, 2}, Slice{0, 3, 2});
	EXPECT_DOUBLE_EQ(reduction::mean(view), -0.5);
}

TEST(NumericMeanEdges, FloatInputIsConvertedBeforeCancellation)
{
	// All inputs are exactly representable in float; a float accumulator loses 1.
	const Vector<float> source{16777216.0F, 1.0F, -16777216.0F};
	EXPECT_DOUBLE_EQ(reduction::mean(source), 1.0 / 3.0);
	EXPECT_DOUBLE_EQ(reduction::mean(Vector<double>{1.0e16, -1.0e16, 1.0}), 1.0 / 3.0);
}

TEST(NumericDivisionEdges, GuardUsesNativeExpressionType)
{
	EXPECT_NO_THROW(require_valid_division(std::int8_t{-128}, std::int8_t{-1}));
	EXPECT_NO_THROW(require_valid_division(std::int16_t{-32768}, std::int16_t{-1}));
	EXPECT_THROW(require_valid_division(std::numeric_limits<int>::min(), -1),
		Exceptions::OverflowError);
	EXPECT_THROW(require_valid_division(std::numeric_limits<std::int64_t>::min(), std::int32_t{-1}),
		Exceptions::OverflowError);
	EXPECT_NO_THROW(require_valid_division(std::numeric_limits<std::int64_t>::min(),
		std::numeric_limits<std::uint64_t>::max()));
	EXPECT_NO_THROW(require_valid_division(std::numeric_limits<std::int32_t>::min(), std::int64_t{-1}));
	EXPECT_NO_THROW(require_valid_division(std::numeric_limits<std::int64_t>::min(), -1.0));
	EXPECT_THROW(require_valid_division(1, 0), Exceptions::ZeroDivisionError);
	EXPECT_THROW(require_valid_division(1.0, -0.0), Exceptions::ZeroDivisionError);
	EXPECT_THROW(require_valid_division(std::complex<double>{1.0, 2.0}, std::complex<double>{}),
		Exceptions::ZeroDivisionError);
}

TEST(NumericDivisionEdges, SignedMinimumIsRejectedInEveryOperandArrangement)
{
	using T = std::int64_t;
	constexpr T lowest = std::numeric_limits<T>::min();
	const Vector<T> numerator{8, lowest};
	const Vector<T> denominator{2, -1};
	EXPECT_THROW((void)(numerator / denominator), Exceptions::OverflowError);
	EXPECT_THROW((void)(numerator / T{-1}), Exceptions::OverflowError);
	EXPECT_THROW((void)(lowest / denominator), Exceptions::OverflowError);
	EXPECT_THROW((void)(Matrix<T>{{8, lowest}, {12, lowest}} / denominator), Exceptions::OverflowError);

	auto array_compound = numerator;
	EXPECT_THROW(array_compound /= denominator, Exceptions::OverflowError);
	EXPECT_EQ(array_compound[0], 8);
	EXPECT_EQ(array_compound[1], lowest);
	auto scalar_compound = numerator;
	EXPECT_THROW(scalar_compound /= T{-1}, Exceptions::OverflowError);
	EXPECT_EQ(scalar_compound[0], 8);
	EXPECT_EQ(scalar_compound[1], lowest);
}

TEST(NumericDivisionEdges, MixedIntegerTypesUseNativeDivisionBeforeResultConversion)
{
	constexpr auto lowest = std::numeric_limits<std::int32_t>::min();
	const Vector<std::int32_t> source{lowest};
	const auto widened = source / std::int64_t{-1};
	static_assert(std::same_as<typename decltype(widened)::value_type, std::int64_t>);
	EXPECT_EQ(widened[0], std::int64_t{2147483648});

	const Vector<std::int64_t> signed_values{std::numeric_limits<std::int64_t>::min()};
	const Vector<std::uint64_t> unsigned_values{std::numeric_limits<std::uint64_t>::max()};
	const auto unsigned_expression = signed_values / unsigned_values;
	static_assert(std::same_as<typename decltype(unsigned_expression)::value_type, double>);
	EXPECT_DOUBLE_EQ(unsigned_expression[0], 0.0);
}

TEST(NumericDivisionEdges, ExhaustiveEightBitPairsMatchIntegerReference)
{
	// Restrict only the quotient's conversion back to the narrow result dtype.
	// Signed min/-1 is valid in the native int expression, but exceeds int8.
	for (int numerator = -128; numerator <= 127; ++numerator)
	{
		for (int denominator = -128; denominator <= 127; ++denominator)
		{
			if (denominator == 0 || (numerator == -128 && denominator == -1))
			{
				continue;
			}
			const Vector<std::int8_t> lhs{static_cast<std::int8_t>(numerator)};
			const Vector<std::int8_t> rhs{static_cast<std::int8_t>(denominator)};
			EXPECT_EQ((lhs / rhs)[0], numerator / denominator)
				<< numerator << " / " << denominator;
		}
	}
}

TEST(NumericArithmeticEdges, GeneratedSafeMixedIntegerCasesMatchWideReference)
{
	const std::array<int, 9> signed_values{-32768, -1024, -17, -1, 0, 1, 17, 1024, 32767};
	const std::array<unsigned, 8> unsigned_values{0, 1, 2, 17, 127, 255, 1024, 65535};
	for (const int left : signed_values)
	{
		for (const unsigned right : unsigned_values)
		{
			const Vector<std::int32_t> lhs{left};
			const Vector<std::uint16_t> rhs{static_cast<std::uint16_t>(right)};
			const auto added = lhs + rhs;
			const auto subtracted = lhs - rhs;
			const auto multiplied = lhs * rhs;
			static_assert(std::same_as<typename decltype(added)::value_type, std::int32_t>);
			EXPECT_EQ(added[0], std::int64_t{left} + right);
			EXPECT_EQ(subtracted[0], std::int64_t{left} - right);
			EXPECT_EQ(multiplied[0], std::int64_t{left} * right);
		}
	}
}

TEST(NumericArithmeticEdges, ResultDtypeDoesNotWidenNativeExpression)
{
	const auto integer_result = Vector<std::int32_t>{-3} + Vector<std::uint32_t>{1};
	static_assert(std::same_as<typename decltype(integer_result)::value_type, std::int64_t>);
	EXPECT_EQ(integer_result[0], std::int64_t{4294967294});
	const auto floating_result = Vector<float>{16777216.0F} + Vector<std::int32_t>{1};
	static_assert(std::same_as<typename decltype(floating_result)::value_type, double>);
	EXPECT_DOUBLE_EQ(floating_result[0], 16777216.0);
}

TEST(NumericFloatingEdges, VarianceResistsLargeOffsetCancellation)
{
	const Vector<double> centered{1.0, 2.0, 3.0, 4.0};
	for (const double offset : {0.0, 1.0e6, 1.0e12, -1.0e12})
	{
		const auto shifted = centered + offset;
		EXPECT_DOUBLE_EQ(reduction::var(shifted), 1.25);
		EXPECT_DOUBLE_EQ(reduction::std(shifted), std::sqrt(1.25));
		EXPECT_DOUBLE_EQ(reduction::mean(shifted), offset + 2.5);
	}
	EXPECT_DOUBLE_EQ(reduction::var(Vector<double>{1.0e150, 1.0e150}), 0.0);
}

TEST(NumericFloatingEdges, NonfiniteValuesFollowDocumentedReductionBehavior)
{
	if (!std::numeric_limits<double>::is_iec559)
	{
		GTEST_SKIP() << "Requires IEC 60559 floating-point behavior.";
	}
	const double infinity = std::numeric_limits<double>::infinity();
	const double nan = std::numeric_limits<double>::quiet_NaN();
	EXPECT_EQ(reduction::mean(Vector<double>{infinity, 1.0}), infinity);
	EXPECT_EQ(reduction::mean(Vector<double>{-infinity, 1.0}), -infinity);
	EXPECT_TRUE(std::isnan(reduction::mean(Vector<double>{infinity, -infinity})));
	EXPECT_TRUE(std::isnan(reduction::mean(Vector<double>{1.0, nan})));
	EXPECT_TRUE(std::isnan(reduction::var(Vector<double>{infinity, 1.0})));
	EXPECT_TRUE(std::isnan(reduction::var(Vector<double>{1.0, nan})));
	EXPECT_TRUE(std::isnan(reduction::std(Vector<double>{1.0, nan})));

	const Vector<double> first_nan{nan, 2.0, 1.0};
	const Vector<double> later_nan{2.0, nan, 1.0};
	EXPECT_TRUE(std::isnan(reduction::min(first_nan)));
	EXPECT_TRUE(std::isnan(reduction::max(first_nan)));
	EXPECT_EQ(reduction::argmin(first_nan), 0);
	EXPECT_EQ(reduction::argmax(first_nan), 0);
	EXPECT_DOUBLE_EQ(reduction::min(later_nan), 1.0);
	EXPECT_DOUBLE_EQ(reduction::max(later_nan), 2.0);
	EXPECT_EQ(reduction::argmin(later_nan), 2);
	EXPECT_EQ(reduction::argmax(later_nan), 0);
}

TEST(NumericFloatingEdges, SignedZeroIsPreservedByDivisionAndFirstExtremaTie)
{
	if (!std::numeric_limits<double>::is_iec559)
	{
		GTEST_SKIP() << "Requires IEC 60559 floating-point behavior.";
	}
	const Vector<double> zeros{-0.0, 0.0};
	const auto divided = zeros / 2.0;
	EXPECT_TRUE(std::signbit(divided[0]));
	EXPECT_FALSE(std::signbit(divided[1]));
	EXPECT_TRUE(std::signbit(reduction::min(zeros)));
	EXPECT_TRUE(std::signbit(reduction::max(zeros)));
	EXPECT_EQ(reduction::argmin(zeros), 0);
	EXPECT_EQ(reduction::argmax(zeros), 0);
	EXPECT_THROW((void)(1.0 / zeros), Exceptions::ZeroDivisionError);
	EXPECT_THROW((void)(zeros / -0.0), Exceptions::ZeroDivisionError);
}
