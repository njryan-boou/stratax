#include <gtest/gtest.h>

#include <complex>
#include <concepts>
#include <type_traits>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(ArrayArithmetic, VectorOperators)
{
	const Vector<int> lhs{12, 18, 24};
	const Vector<int> rhs{2, 3, 4};
	const auto sums = lhs + rhs;
	const auto differences = lhs - rhs;
	const auto products = lhs * rhs;
	const auto quotients = lhs / rhs;

	EXPECT_EQ(std::vector<int>(sums.begin(), sums.end()),
		(std::vector<int>{14, 21, 28}));
	EXPECT_EQ(std::vector<int>(differences.begin(), differences.end()),
		(std::vector<int>{10, 15, 20}));
	EXPECT_EQ(std::vector<int>(products.begin(), products.end()),
		(std::vector<int>{24, 54, 96}));
	EXPECT_EQ(std::vector<int>(quotients.begin(), quotients.end()),
		(std::vector<int>{6, 6, 6}));
}

TEST(ArrayArithmetic, MatrixBroadcasting)
{
	const Matrix<int> column{{1}, {10}};
	const Matrix<int> row{{1, 2, 3}};
	const auto result = column + row;

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Matrix<int>>);
	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{2, 3, 4, 11, 12, 13}));
}

TEST(ArrayArithmetic, TensorBroadcasting)
{
	Tensor<int> lhs(Shape{2, 1, 2});
	Tensor<int> rhs(Shape{1, 3, 1});
	lhs[0] = 1;
	lhs[1] = 2;
	lhs[2] = 3;
	lhs[3] = 4;
	rhs[0] = 10;
	rhs[1] = 20;
	rhs[2] = 30;

	const auto result = lhs * rhs;

	EXPECT_EQ(result.shape(), Shape({2, 3, 2}));
	EXPECT_EQ(result(0, 0, 0), 10);
	EXPECT_EQ(result(0, 2, 1), 60);
	EXPECT_EQ(result(1, 0, 0), 30);
	EXPECT_EQ(result(1, 2, 1), 120);
}

TEST(ArrayArithmetic, DoesNotModifyOperands)
{
	const Matrix<int> lhs{{1, 2}, {3, 4}};
	const Matrix<int> rhs{{10, 20}};
	const auto result = lhs + rhs;

	EXPECT_NE(result.data(), lhs.data());
	EXPECT_NE(result.data(), rhs.data());
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()),
		(std::vector<int>{1, 2, 3, 4}));
	EXPECT_EQ(std::vector<int>(rhs.begin(), rhs.end()),
		(std::vector<int>{10, 20}));
}

TEST(ArrayArithmetic, RejectsIncompatibleShapes)
{
	const Matrix<int> lhs(2, 3);
	const Matrix<int> rhs(4, 3);

	EXPECT_THROW(static_cast<void>(lhs + rhs), Exceptions::BroadcastError);
	EXPECT_THROW(static_cast<void>(lhs - rhs), Exceptions::BroadcastError);
	EXPECT_THROW(static_cast<void>(lhs * rhs), Exceptions::BroadcastError);
	EXPECT_THROW(static_cast<void>(lhs / rhs), Exceptions::BroadcastError);
}

TEST(ArrayScalarArithmetic, ArrayOnLeft)
{
	const Vector<int> source{10, 20, 30};

	const auto added = source + 2;
	const auto subtracted = source - 2;
	const auto multiplied = source * 2;
	const auto divided = source / 2;

	EXPECT_EQ(std::vector<int>(added.begin(), added.end()),
		(std::vector<int>{12, 22, 32}));
	EXPECT_EQ(std::vector<int>(subtracted.begin(), subtracted.end()),
		(std::vector<int>{8, 18, 28}));
	EXPECT_EQ(std::vector<int>(multiplied.begin(), multiplied.end()),
		(std::vector<int>{20, 40, 60}));
	EXPECT_EQ(std::vector<int>(divided.begin(), divided.end()),
		(std::vector<int>{5, 10, 15}));
}

TEST(ArrayScalarArithmetic, ScalarOnLeftPreservesOperandOrder)
{
	const Vector<int> source{2, 4, 8};

	const auto added = 16 + source;
	const auto subtracted = 16 - source;
	const auto multiplied = 16 * source;
	const auto divided = 16 / source;

	EXPECT_EQ(std::vector<int>(added.begin(), added.end()),
		(std::vector<int>{18, 20, 24}));
	EXPECT_EQ(std::vector<int>(subtracted.begin(), subtracted.end()),
		(std::vector<int>{14, 12, 8}));
	EXPECT_EQ(std::vector<int>(multiplied.begin(), multiplied.end()),
		(std::vector<int>{32, 64, 128}));
	EXPECT_EQ(std::vector<int>(divided.begin(), divided.end()),
		(std::vector<int>{8, 4, 2}));
}

TEST(ArrayScalarArithmetic, PreservesShapeAndContainerType)
{
	const Tensor<double> source(Shape{2, 1, 3}, 1.5);
	const auto result = source + 2.0;

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Tensor<double>>);
	EXPECT_EQ(result.shape(), source.shape());
	for (double value : result) {
		EXPECT_DOUBLE_EQ(value, 3.5);
	}
}

TEST(ArrayArithmetic, SupportsComplexValues)
{
	using Complex = std::complex<double>;
	const Vector<Complex> lhs{Complex(1.0, 2.0), Complex(3.0, -1.0)};
	const Vector<Complex> rhs{Complex(2.0, 0.0), Complex(0.0, 1.0)};

	const auto sum = lhs + rhs;
	const auto product = lhs * rhs;

	EXPECT_EQ(sum[0], Complex(3.0, 2.0));
	EXPECT_EQ(sum[1], Complex(3.0, 0.0));
	EXPECT_EQ(product[0], Complex(2.0, 4.0));
	EXPECT_EQ(product[1], Complex(1.0, 3.0));
}

TEST(ArrayDivision, RejectsZeroArrayElement)
{
	const Matrix<int> lhs{{10, 20}, {30, 40}};
	const Matrix<int> rhs{{2, 0}};

	EXPECT_THROW(static_cast<void>(lhs / rhs), Exceptions::ZeroDivisionError);
}

TEST(ArrayDivision, ArrayDivisorErrorMessage)
{
	const Vector<int> lhs{1, 2};
	const Vector<int> rhs{1, 0};

	try {
		static_cast<void>(lhs / rhs);
		FAIL() << "Expected Exceptions::ZeroDivisionError";
	} catch (const Exceptions::ZeroDivisionError& error) {
		EXPECT_STREQ(error.what(),
			"Array division divisor element cannot be zero.");
	}
}

TEST(ArrayDivision, RejectsZeroRightScalar)
{
	const Vector<int> source{1, 2, 3};

	EXPECT_THROW(static_cast<void>(source / 0), Exceptions::ZeroDivisionError);
}

TEST(ArrayDivision, RightScalarErrorMessage)
{
	const Vector<int> source{1};

	try {
		static_cast<void>(source / 0);
		FAIL() << "Expected Exceptions::ZeroDivisionError";
	} catch (const Exceptions::ZeroDivisionError& error) {
		EXPECT_STREQ(error.what(),
			"Array division scalar divisor cannot be zero.");
	}
}

TEST(ArrayDivision, RejectsZeroArrayDivisorForLeftScalar)
{
	const Vector<int> source{1, 0, 2};

	EXPECT_THROW(static_cast<void>(10 / source), Exceptions::ZeroDivisionError);
}

TEST(ArrayDivision, LeftScalarErrorMessage)
{
	const Vector<int> source{0};

	try {
		static_cast<void>(1 / source);
		FAIL() << "Expected Exceptions::ZeroDivisionError";
	} catch (const Exceptions::ZeroDivisionError& error) {
		EXPECT_STREQ(error.what(),
			"Scalar division divisor element cannot be zero.");
	}
}

TEST(CompoundArrayArithmetic, AllOperatorsReturnLeftReference)
{
	Vector<int> value{24, 36, 48};
	const Vector<int> rhs{2, 3, 4};

	EXPECT_EQ(&(value += rhs), &value);
	EXPECT_EQ(&(value -= rhs), &value);
	EXPECT_EQ(&(value *= rhs), &value);
	EXPECT_EQ(&(value /= rhs), &value);
	EXPECT_EQ(std::vector<int>(value.begin(), value.end()),
		(std::vector<int>{24, 36, 48}));
}

TEST(CompoundArrayArithmetic, CanAcquireBroadcastedShape)
{
	Matrix<int> lhs{{1}, {10}};
	const Matrix<int> rhs{{1, 2, 3}};

	lhs += rhs;

	EXPECT_EQ(lhs.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()),
		(std::vector<int>{2, 3, 4, 11, 12, 13}));
}

TEST(CompoundScalarArithmetic, AllOperatorsReturnLeftReference)
{
	Vector<int> value{8, 16, 24};

	EXPECT_EQ(&(value += 2), &value);
	EXPECT_EQ(&(value -= 2), &value);
	EXPECT_EQ(&(value *= 2), &value);
	EXPECT_EQ(&(value /= 2), &value);
	EXPECT_EQ(std::vector<int>(value.begin(), value.end()),
		(std::vector<int>{8, 16, 24}));
}

TEST(CompoundArithmetic, FailureLeavesLeftOperandUnchanged)
{
	Matrix<int> lhs{{10, 20}, {30, 40}};
	const Matrix<int> original = lhs;
	const Matrix<int> zero_divisor{{1, 0}};

	EXPECT_THROW(lhs /= zero_divisor, Exceptions::ZeroDivisionError);
	EXPECT_EQ(lhs.shape(), original.shape());
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()),
		std::vector<int>(original.begin(), original.end()));

	const Matrix<int> incompatible(3, 4);
	EXPECT_THROW(lhs += incompatible, Exceptions::BroadcastError);
	EXPECT_EQ(lhs.shape(), original.shape());
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()),
		std::vector<int>(original.begin(), original.end()));
}

TEST(UnaryArithmetic, MinusNegatesValues)
{
	const Matrix<int> source{{-2, 0}, {3, -4}};
	const auto result = -source;

	EXPECT_EQ(result.shape(), source.shape());
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{2, 0, -3, 4}));
}

TEST(UnaryArithmetic, PlusReturnsIndependentCopy)
{
	Vector<int> source{1, 2, 3};
	auto result = +source;

	EXPECT_NE(result.data(), source.data());
	result[0] = 99;
	EXPECT_EQ(source[0], 1);
}

TEST(EmptyArithmetic, ArrayAndScalarOperationsPreserveShape)
{
	const Tensor<int> lhs(Shape{2, 0, 3});
	const Tensor<int> rhs(Shape{1, 0, 3});

	EXPECT_EQ((lhs + rhs).shape(), Shape({2, 0, 3}));
	EXPECT_EQ((lhs - 2).shape(), lhs.shape());
	EXPECT_EQ((2 * lhs).shape(), lhs.shape());
	EXPECT_EQ((-lhs).shape(), lhs.shape());
}

TEST(EmptyArithmetic, ArrayDivisionHasNoUsedDivisors)
{
	const Vector<int> lhs;
	const Vector<int> rhs;

	EXPECT_TRUE((lhs / rhs).empty());
	EXPECT_TRUE((1 / rhs).empty());
}

TEST(EmptyArithmetic, ZeroScalarDivisionStillThrows)
{
	const Vector<int> source;

	EXPECT_THROW(static_cast<void>(source / 0), Exceptions::ZeroDivisionError);
}
