#include <gtest/gtest.h>

#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;
using namespace stratax::creation;

TEST(CreationZeros, Tensor)
{
	const auto result = zeros<int>(Shape{2, 3, 4});

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Tensor<int>>);
	EXPECT_EQ(result.shape(), Shape({2, 3, 4}));
	for (const int value : result) {
		EXPECT_EQ(value, 0);
	}
}

TEST(CreationZeros, Vector)
{
	const auto result = zeros<double>(5);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Vector<double>>);
	EXPECT_EQ(result.shape(), Shape({5}));
	for (const double value : result) {
		EXPECT_EQ(value, 0.0);
	}
}

TEST(CreationZeros, Matrix)
{
	const auto result = zeros<float>(2, 3);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Matrix<float>>);
	EXPECT_EQ(result.shape(), Shape({2, 3}));
	for (const float value : result) {
		EXPECT_EQ(value, 0.0F);
	}
}

TEST(CreationZeros, RankZeroTensor)
{
	const auto result = zeros<int>(Shape{});

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.rank(), 0);
	EXPECT_EQ(result.shape(), Shape{});
}

TEST(CreationZeros, ZeroSizedVector)
{
	const auto result = zeros<int>(0);

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({0}));
}

TEST(CreationZeros, ZeroSizedMatrixDimensions)
{
	const auto zero_rows = zeros<int>(0, 4);
	const auto zero_cols = zeros<int>(3, 0);

	EXPECT_TRUE(zero_rows.empty());
	EXPECT_EQ(zero_rows.shape(), Shape({0, 4}));
	EXPECT_TRUE(zero_cols.empty());
	EXPECT_EQ(zero_cols.shape(), Shape({3, 0}));
}

TEST(CreationOnes, Tensor)
{
	const auto result = ones<int>(Shape{2, 3, 4});

	EXPECT_EQ(result.shape(), Shape({2, 3, 4}));
	for (const int value : result) {
		EXPECT_EQ(value, 1);
	}
}

TEST(CreationOnes, Vector)
{
	const auto result = ones<double>(5);

	EXPECT_EQ(result.shape(), Shape({5}));
	for (const double value : result) {
		EXPECT_EQ(value, 1.0);
	}
}

TEST(CreationOnes, Matrix)
{
	const auto result = ones<float>(2, 3);

	EXPECT_EQ(result.shape(), Shape({2, 3}));
	for (const float value : result) {
		EXPECT_EQ(value, 1.0F);
	}
}

TEST(CreationOnes, ComplexValues)
{
	using Complex = std::complex<double>;
	const auto tensor = ones<Complex>(Shape{2, 2});
	const auto vector = ones<Complex>(3);
	const auto matrix = ones<Complex>(2, 2);

	for (const Complex value : tensor) {
		EXPECT_EQ(value, Complex(1.0, 0.0));
	}
	for (const Complex value : vector) {
		EXPECT_EQ(value, Complex(1.0, 0.0));
	}
	for (const Complex value : matrix) {
		EXPECT_EQ(value, Complex(1.0, 0.0));
	}
}

TEST(CreationFull, Tensor)
{
	const auto result = full<int>(Shape{2, 3, 4}, -7);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Tensor<int>>);
	EXPECT_EQ(result.shape(), Shape({2, 3, 4}));
	for (const int value : result) {
		EXPECT_EQ(value, -7);
	}
}

TEST(CreationFull, Vector)
{
	const auto result = full<double>(5, 2.5);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Vector<double>>);
	EXPECT_EQ(result.shape(), Shape({5}));
	for (const double value : result) {
		EXPECT_EQ(value, 2.5);
	}
}

TEST(CreationFull, Matrix)
{
	const auto result = full<float>(2, 3, -1.5F);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Matrix<float>>);
	EXPECT_EQ(result.shape(), Shape({2, 3}));
	for (const float value : result) {
		EXPECT_EQ(value, -1.5F);
	}
}

TEST(CreationFull, ZeroSizedResultsPreserveShape)
{
	const auto tensor = full<int>(Shape{2, 0, 4}, 7);
	const auto vector = full<int>(0, 7);
	const auto matrix = full<int>(3, 0, 7);

	EXPECT_TRUE(tensor.empty());
	EXPECT_EQ(tensor.shape(), Shape({2, 0, 4}));
	EXPECT_TRUE(vector.empty());
	EXPECT_EQ(vector.shape(), Shape({0}));
	EXPECT_TRUE(matrix.empty());
	EXPECT_EQ(matrix.shape(), Shape({3, 0}));
}

TEST(CreationIdentity, Values)
{
	const auto result = identity<int>(4);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Matrix<int>>);
	EXPECT_EQ(result.shape(), Shape({4, 4}));

	for (std::size_t row = 0; row < result.rows(); ++row) {
		for (std::size_t col = 0; col < result.cols(); ++col) {
			EXPECT_EQ(result(row, col), row == col ? 1 : 0);
		}
	}
}

TEST(CreationIdentity, ZeroSize)
{
	const auto result = identity<double>(0);

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({0, 0}));
}

TEST(CreationIdentity, ComplexValues)
{
	using Complex = std::complex<double>;
	const auto result = identity<Complex>(3);

	for (std::size_t row = 0; row < result.rows(); ++row) {
		for (std::size_t col = 0; col < result.cols(); ++col) {
			const Complex expected = row == col ? Complex(1.0, 0.0) : Complex{};
			EXPECT_EQ(result(row, col), expected);
		}
	}
}

TEST(CreationOverflow, TensorElementCount)
{
	EXPECT_THROW(
		static_cast<void>(zeros<int>(
			Shape{std::numeric_limits<std::size_t>::max(), 2})),
		Exceptions::DimensionError
	);
}

TEST(CreationOverflow, MatrixElementCount)
{
	EXPECT_THROW(
		static_cast<void>(ones<int>(
			std::numeric_limits<std::size_t>::max(), 2)),
		Exceptions::DimensionError
	);
}

TEST(CreationOverflow, IdentityElementCount)
{
	EXPECT_THROW(
		static_cast<void>(identity<int>(
			std::numeric_limits<std::size_t>::max())),
		Exceptions::DimensionError
	);
}
