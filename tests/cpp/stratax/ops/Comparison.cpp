#include <gtest/gtest.h>

#include <complex>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(ArrayComparison, EqualVectors)
{
	const Vector<int> lhs{1, 2, 3};
	const Vector<int> rhs{1, 2, 3};

	EXPECT_TRUE(lhs == rhs);
	EXPECT_FALSE(lhs != rhs);
}

TEST(ArrayComparison, DetectsValueDifference)
{
	const Matrix<int> lhs{{1, 2}, {3, 4}};
	const Matrix<int> rhs{{1, 2}, {3, 5}};

	EXPECT_FALSE(lhs == rhs);
	EXPECT_TRUE(lhs != rhs);
}

TEST(ArrayComparison, ShapeSensitiveWithEqualFlatValues)
{
	const Tensor<int> lhs(Shape{2, 2});
	const Tensor<int> rhs(Shape{4});

	EXPECT_FALSE(lhs == rhs);
	EXPECT_TRUE(lhs != rhs);
}

TEST(ArrayComparison, DoesNotBroadcast)
{
	const Matrix<int> lhs{{1, 2, 3}, {1, 2, 3}};
	const Matrix<int> rhs{{1, 2, 3}};

	EXPECT_FALSE(lhs == rhs);
	EXPECT_TRUE(lhs != rhs);
}

TEST(ArrayComparison, EqualMatricesAndTensors)
{
	const Matrix<double> matrix_lhs{{1.0, 2.0}, {3.0, 4.0}};
	const Matrix<double> matrix_rhs{{1.0, 2.0}, {3.0, 4.0}};
	const Tensor<int> tensor_lhs(Shape{2, 1, 3}, 7);
	const Tensor<int> tensor_rhs(Shape{2, 1, 3}, 7);

	EXPECT_TRUE(matrix_lhs == matrix_rhs);
	EXPECT_TRUE(tensor_lhs == tensor_rhs);
}

TEST(ArrayComparison, UsesExactFloatingPointEquality)
{
	const Vector<double> lhs{0.1 + 0.2};
	const Vector<double> rhs{0.3};

	EXPECT_TRUE(lhs != rhs);
}

TEST(ArrayComparison, SupportsComplexValues)
{
	using Complex = std::complex<double>;
	const Vector<Complex> lhs{Complex(1.0, 2.0), Complex(3.0, 4.0)};
	const Vector<Complex> equal{Complex(1.0, 2.0), Complex(3.0, 4.0)};
	const Vector<Complex> different{Complex(1.0, 2.0), Complex(3.0, -4.0)};

	EXPECT_TRUE(lhs == equal);
	EXPECT_TRUE(lhs != different);
}

TEST(ArrayComparison, EmptyArraysRequireEqualShapes)
{
	const Tensor<int> lhs(Shape{2, 0, 3});
	const Tensor<int> equal(Shape{2, 0, 3});
	const Tensor<int> different(Shape{0, 6});

	EXPECT_TRUE(lhs == equal);
	EXPECT_TRUE(lhs != different);
}

TEST(ArrayComparison, ReflexiveForOrdinaryValues)
{
	const Vector<int> source{-3, 0, 7};

	EXPECT_TRUE(source == source);
	EXPECT_FALSE(source != source);
}
