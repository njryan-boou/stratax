#include <gtest/gtest.h>

#include <concepts>
#include <complex>
#include <type_traits>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;
namespace dtype = stratax::dtype;

template<typename A>
concept LessComparableArray = requires(const A& lhs, const A& rhs)
{
	lhs < rhs;
};

TEST(ArrayComparison, NamedFunctionsCompareEveryRelation)
{
	const Vector<int> lhs{1, 2, 3};
	const Vector<int> rhs{1, 3, 2};

	EXPECT_TRUE(equal(lhs, rhs)[0]);
	EXPECT_TRUE(not_equal(lhs, rhs)[1]);
	EXPECT_TRUE(less(lhs, rhs)[1]);
	EXPECT_TRUE(less_equal(lhs, rhs)[0]);
	EXPECT_TRUE(greater(lhs, rhs)[2]);
	EXPECT_TRUE(greater_equal(lhs, rhs)[0]);
}

TEST(ArrayComparison, OperatorsCompareEveryRelation)
{
	const Vector<int> lhs{1, 2, 3};
	const Vector<int> rhs{1, 3, 2};

	EXPECT_TRUE((lhs == rhs)[0]);
	EXPECT_TRUE((lhs != rhs)[1]);
	EXPECT_TRUE((lhs < rhs)[1]);
	EXPECT_TRUE((lhs <= rhs)[0]);
	EXPECT_TRUE((lhs > rhs)[2]);
	EXPECT_TRUE((lhs >= rhs)[0]);
}

TEST(ArrayComparison, ComplexSupportsEqualityButNotOrdering)
{
	using complex_type = std::complex<double>;
	const Vector<complex_type> lhs{{1.0, 2.0}, {3.0, 4.0}};
	const Vector<complex_type> rhs{{1.0, 2.0}, {3.0, -4.0}};

	static_assert(!LessComparableArray<Vector<complex_type>>);
	EXPECT_TRUE((lhs == rhs)[0]);
	EXPECT_TRUE((lhs != rhs)[1]);
}

TEST(ArrayComparison, MixedContainersBroadcastAndReturnBooleanTensor)
{
	const Vector<dtype::int16> lhs{1, 2, 3};
	const Matrix<dtype::float32> rhs{{1.0F}, {2.5F}};
	const auto result = lhs < rhs;

	static_assert(std::same_as<
		std::remove_cv_t<decltype(result)>,
		Tensor<dtype::bool_>>);
	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_FALSE(result(0, 0));
	EXPECT_FALSE(result(0, 1));
	EXPECT_FALSE(result(0, 2));
	EXPECT_TRUE(result(1, 0));
	EXPECT_TRUE(result(1, 1));
	EXPECT_FALSE(result(1, 2));
}

TEST(ArrayScalarComparison, NamedFunctionsSupportBothOperandOrders)
{
	const Vector<int> values{1, 2, 3};

	const auto array_first = less(values, 3);
	const auto scalar_first = less(2, values);
	const auto less_or_equal = less_equal(2, values);
	const auto greater_result = greater(values, 1);
	const auto greater_or_equal = greater_equal(2, values);

	EXPECT_TRUE(array_first[0]);
	EXPECT_TRUE(array_first[1]);
	EXPECT_FALSE(array_first[2]);
	EXPECT_FALSE(scalar_first[0]);
	EXPECT_FALSE(scalar_first[1]);
	EXPECT_TRUE(scalar_first[2]);
	EXPECT_FALSE(less_or_equal[0]);
	EXPECT_TRUE(less_or_equal[1]);
	EXPECT_TRUE(less_or_equal[2]);
	EXPECT_FALSE(greater_result[0]);
	EXPECT_TRUE(greater_result[1]);
	EXPECT_TRUE(greater_result[2]);
	EXPECT_TRUE(greater_or_equal[0]);
	EXPECT_TRUE(greater_or_equal[1]);
	EXPECT_FALSE(greater_or_equal[2]);
}

TEST(ArrayScalarComparison, OperatorsSupportArrayOnLeft)
{
	const Vector<int> values{1, 2, 3};

	EXPECT_TRUE((values == 2)[1]);
	EXPECT_TRUE((values != 2)[0]);
	EXPECT_TRUE((values < 2)[0]);
	EXPECT_TRUE((values <= 2)[1]);
	EXPECT_TRUE((values > 2)[2]);
	EXPECT_TRUE((values >= 2)[1]);
}

TEST(ArrayScalarComparison, OperatorsSupportScalarOnLeft)
{
	const Vector<int> values{1, 2, 3};

	EXPECT_TRUE((2 == values)[1]);
	EXPECT_TRUE((2 != values)[0]);
	EXPECT_TRUE((2 < values)[2]);
	EXPECT_TRUE((2 <= values)[1]);
	EXPECT_TRUE((2 > values)[0]);
	EXPECT_FALSE((2 > values)[1]);
	EXPECT_TRUE((2 >= values)[1]);
	EXPECT_FALSE((2 >= values)[2]);
}

TEST(ArrayScalarComparison, BooleanScalarsAreDTypes)
{
	const Vector<dtype::bool_> values{false, true};

	const auto equal_true = values == true;
	const auto false_less = false < values;

	static_assert(std::same_as<
		std::remove_cv_t<decltype(equal_true)>,
		Vector<dtype::bool_>>);
	EXPECT_FALSE(equal_true[0]);
	EXPECT_TRUE(equal_true[1]);
	EXPECT_FALSE(false_less[0]);
	EXPECT_TRUE(false_less[1]);
}

TEST(ArrayComparison, RejectsIncompatibleShapes)
{
	const Matrix<int> lhs(2, 3);
	const Matrix<int> rhs(4, 3);

	EXPECT_THROW(static_cast<void>(lhs == rhs), Exceptions::BroadcastError);
	EXPECT_THROW(static_cast<void>(lhs < rhs), Exceptions::BroadcastError);
}

TEST(ArrayComparison, EmptyResultsPreserveBroadcastedShape)
{
	const Tensor<int> lhs(Shape{2, 0, 3});
	const Tensor<int> rhs(Shape{1, 0, 3});
	const auto result = lhs == rhs;

	EXPECT_EQ(result.shape(), Shape({2, 0, 3}));
	EXPECT_TRUE(result.empty());
}
