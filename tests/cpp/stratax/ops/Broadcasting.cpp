#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(Broadcastable, EqualShapes)
{
	EXPECT_TRUE(broadcastable(Shape{2, 3, 4}, Shape{2, 3, 4}));
}

TEST(Broadcastable, AlignsTrailingDimensions)
{
	EXPECT_TRUE(broadcastable(Shape{2, 3, 4}, Shape{4}));
	EXPECT_TRUE(broadcastable(Shape{2, 3, 4}, Shape{3, 4}));
	EXPECT_TRUE(broadcastable(Shape{3, 1}, Shape{2, 1, 4}));
}

TEST(Broadcastable, SingletonDimensions)
{
	EXPECT_TRUE(broadcastable(Shape{2, 1, 4}, Shape{1, 3, 1}));
	EXPECT_TRUE(broadcastable(Shape{1}, Shape{5, 6, 7}));
}

TEST(Broadcastable, RejectsMismatchedDimensions)
{
	EXPECT_FALSE(broadcastable(Shape{2, 3}, Shape{2, 4}));
	EXPECT_FALSE(broadcastable(Shape{2, 3, 4}, Shape{3, 5}));
}

TEST(Broadcastable, ZeroDimensions)
{
	EXPECT_TRUE(broadcastable(Shape{0, 3}, Shape{1, 3}));
	EXPECT_TRUE(broadcastable(Shape{0, 3}, Shape{0, 3}));
	EXPECT_FALSE(broadcastable(Shape{0, 3}, Shape{2, 3}));
}

TEST(Broadcastable, RankZeroShapeActsAsMissingDimensions)
{
	EXPECT_TRUE(broadcastable(Shape{}, Shape{}));
	EXPECT_TRUE(broadcastable(Shape{}, Shape{2, 3}));
}

TEST(BroadcastedShape, SelectsNonSingletonExtents)
{
	EXPECT_EQ(
		broadcasted_shape(Shape{2, 1, 4}, Shape{1, 3, 1}),
		Shape({2, 3, 4}));
}

TEST(BroadcastedShape, PadsLowerRankShapeOnLeft)
{
	EXPECT_EQ(broadcasted_shape(Shape{3, 1}, Shape{2, 1, 4}),
		Shape({2, 3, 4}));
}

TEST(BroadcastedShape, IsSymmetric)
{
	const Shape left{5, 1, 7};
	const Shape right{1, 3, 1};

	EXPECT_EQ(broadcasted_shape(left, right), broadcasted_shape(right, left));
}

TEST(BroadcastedShape, PreservesZeroExtent)
{
	EXPECT_EQ(broadcasted_shape(Shape{0, 3}, Shape{1, 3}), Shape({0, 3}));
}

TEST(BroadcastedShape, RankZeroPairs)
{
	EXPECT_EQ(broadcasted_shape(Shape{}, Shape{}), Shape{});
	EXPECT_EQ(broadcasted_shape(Shape{}, Shape{2, 3}), Shape({2, 3}));
}

TEST(BroadcastedShape, RejectsIncompatibleShapes)
{
	EXPECT_THROW(
		static_cast<void>(broadcasted_shape(Shape{2, 3}, Shape{4, 3})),
		Exceptions::BroadcastError);
}

TEST(BroadcastedShape, ErrorMessage)
{
	try {
		static_cast<void>(broadcasted_shape(Shape{2, 3}, Shape{2, 4}));
		FAIL() << "Expected Exceptions::BroadcastError";
	} catch (const Exceptions::BroadcastError& error) {
		EXPECT_STREQ(error.what(), "Shapes are not broadcastable");
	}
}

TEST(BroadcastIndex, RepeatsLowerRankOperand)
{
	const Shape result_shape{2, 3};
	const Shape operand_shape{3};

	EXPECT_EQ(broadcast_detail::flat_operand_index(0, result_shape, operand_shape), 0);
	EXPECT_EQ(broadcast_detail::flat_operand_index(2, result_shape, operand_shape), 2);
	EXPECT_EQ(broadcast_detail::flat_operand_index(3, result_shape, operand_shape), 0);
	EXPECT_EQ(broadcast_detail::flat_operand_index(5, result_shape, operand_shape), 2);
}

TEST(BroadcastIndex, CollapsesSingletonDimension)
{
	const Shape result_shape{2, 3};
	const Shape operand_shape{2, 1};

	EXPECT_EQ(broadcast_detail::flat_operand_index(0, result_shape, operand_shape), 0);
	EXPECT_EQ(broadcast_detail::flat_operand_index(2, result_shape, operand_shape), 0);
	EXPECT_EQ(broadcast_detail::flat_operand_index(3, result_shape, operand_shape), 1);
	EXPECT_EQ(broadcast_detail::flat_operand_index(5, result_shape, operand_shape), 1);
}

TEST(BroadcastedArrayOperation, MatchingVectors)
{
	const Vector<int> lhs{1, 2, 3};
	const Vector<int> rhs{10, 20, 30};
	const auto result = broadcasted_op(lhs, rhs,
		[](int left, int right) { return left + right; });

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Vector<int>>);
	EXPECT_EQ(result.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{11, 22, 33}));
}

TEST(BroadcastedArrayOperation, BroadcastsMatrixRow)
{
	const Matrix<int> lhs{{1, 2, 3}, {4, 5, 6}};
	const Matrix<int> rhs{{10, 20, 30}};
	const auto result = broadcasted_op(lhs, rhs,
		[](int left, int right) { return left + right; });

	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{11, 22, 33, 14, 25, 36}));
}

TEST(BroadcastedArrayOperation, BroadcastsMatrixColumn)
{
	const Matrix<int> lhs{{1}, {10}};
	const Matrix<int> rhs{{1, 2, 3}};
	const auto result = broadcasted_op(lhs, rhs,
		[](int left, int right) { return left * right; });

	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, 2, 3, 10, 20, 30}));
}

TEST(BroadcastedArrayOperation, BroadcastsTensorAcrossMultipleAxes)
{
	Tensor<int> lhs(Shape{2, 1, 3});
	Tensor<int> rhs(Shape{1, 4, 1});
	for (std::size_t i = 0; i < lhs.size(); ++i) {
		lhs[i] = static_cast<int>(i + 1);
	}
	for (std::size_t i = 0; i < rhs.size(); ++i) {
		rhs[i] = static_cast<int>((i + 1) * 10);
	}

	const auto result = broadcasted_op(lhs, rhs,
		[](int left, int right) { return left + right; });

	EXPECT_EQ(result.shape(), Shape({2, 4, 3}));
	EXPECT_EQ(result(0, 0, 0), 11);
	EXPECT_EQ(result(0, 3, 2), 43);
	EXPECT_EQ(result(1, 0, 0), 14);
	EXPECT_EQ(result(1, 3, 2), 46);
}

TEST(BroadcastedArrayOperation, ReturnsIndependentStorage)
{
	Matrix<int> lhs{{1, 2}, {3, 4}};
	const Matrix<int> rhs{{10, 20}};
	auto result = broadcasted_op(lhs, rhs,
		[](int left, int right) { return left + right; });

	EXPECT_NE(result.data(), lhs.data());
	EXPECT_NE(result.data(), rhs.data());
	result[0] = 99;
	EXPECT_EQ(lhs[0], 1);
}

TEST(BroadcastedArrayOperation, RejectsIncompatibleShapes)
{
	const Matrix<int> lhs(2, 3);
	const Matrix<int> rhs(2, 4);

	EXPECT_THROW(
		broadcasted_op(lhs, rhs, [](int left, int right) { return left + right; }),
		Exceptions::BroadcastError);
}

TEST(BroadcastedScalarOperation, ArrayOnLeft)
{
	const Vector<int> lhs{1, 2, 3};
	const auto result = broadcasted_op(lhs, 10,
		[](int left, int right) { return left - right; });

	EXPECT_EQ(result.shape(), lhs.shape());
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{-9, -8, -7}));
}

TEST(BroadcastedScalarOperation, ScalarOnLeft)
{
	const Vector<int> rhs{1, 2, 4};
	const auto result = broadcasted_op(8, rhs,
		[](int left, int right) { return left / right; });

	EXPECT_EQ(result.shape(), rhs.shape());
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{8, 4, 2}));
}

TEST(BroadcastedScalarOperation, PreservesArrayContainer)
{
	const Matrix<double> source{{1.0, 2.0}, {3.0, 4.0}};
	const auto right = broadcasted_op(source, 2.0,
		[](double value, double scalar) { return value * scalar; });
	const auto left = broadcasted_op(2.0, source,
		[](double scalar, double value) { return scalar * value; });

	static_assert(std::same_as<std::remove_cv_t<decltype(right)>, Matrix<double>>);
	static_assert(std::same_as<std::remove_cv_t<decltype(left)>, Matrix<double>>);
	EXPECT_EQ(right.shape(), source.shape());
	EXPECT_EQ(left.shape(), source.shape());
}

TEST(BroadcastedOperation, EmptyResultDoesNotInvokeCallable)
{
	const Tensor<int> lhs(Shape{0, 3});
	const Tensor<int> rhs(Shape{1, 3});
	std::size_t calls = 0;

	const auto array_result = broadcasted_op(lhs, rhs,
		[&calls](int left, int right) {
			++calls;
			return left + right;
		});
	const auto scalar_result = broadcasted_op(lhs, 2,
		[&calls](int left, int right) {
			++calls;
			return left * right;
		});

	EXPECT_TRUE(array_result.empty());
	EXPECT_EQ(array_result.shape(), Shape({0, 3}));
	EXPECT_TRUE(scalar_result.empty());
	EXPECT_EQ(scalar_result.shape(), Shape({0, 3}));
	EXPECT_EQ(calls, 0);
}
