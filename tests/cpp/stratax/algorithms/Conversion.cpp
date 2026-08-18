#include <gtest/gtest.h>

#include <complex>
#include <concepts>
#include <type_traits>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::conversion;
using namespace stratax::core;

TEST(ToVector, RankOneVector)
{
	const Vector<int> source{1, 2, 3, 4};
	const auto result = to_vector(source);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Vector<int>>);
	EXPECT_EQ(result.shape(), Shape({4}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, 2, 3, 4}));
}

TEST(ToVector, RowAndColumnMatrices)
{
	const Matrix<int> row{{1, 2, 3}};
	const Matrix<int> column{{1}, {2}, {3}};

	const auto row_result = to_vector(row);
	const auto column_result = to_vector(column);

	EXPECT_EQ(row_result.shape(), Shape({3}));
	EXPECT_EQ(column_result.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(row_result.begin(), row_result.end()),
		(std::vector<int>{1, 2, 3}));
	EXPECT_EQ(std::vector<int>(column_result.begin(), column_result.end()),
		(std::vector<int>{1, 2, 3}));
}

TEST(ToVector, RemovesSingletonDimensions)
{
	Tensor<int> source(Shape{1, 1, 4, 1});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i + 1);
	}

	const auto result = to_vector(source);

	EXPECT_EQ(result.shape(), Shape({4}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, 2, 3, 4}));
}

TEST(ToVector, PreservesFlatOrder)
{
	const Matrix<int> source{{1, 2, 3, 4}};
	const auto result = to_vector(source);

	EXPECT_EQ(result[0], 1);
	EXPECT_EQ(result[1], 2);
	EXPECT_EQ(result[2], 3);
	EXPECT_EQ(result[3], 4);
}

TEST(ToVector, ReturnsIndependentStorage)
{
	Matrix<int> source{{1, 2, 3}};
	auto result = to_vector(source);

	EXPECT_NE(result.data(), source.data());
	result[0] = 99;
	EXPECT_EQ(source[0], 1);
}

TEST(ToVector, RejectsMultipleNonSingletonDimensions)
{
	const Matrix<int> source{{1, 2}, {3, 4}};

	EXPECT_THROW(static_cast<void>(to_vector(source)), Exceptions::ShapeError);
}

TEST(ToVector, ShapeErrorMessage)
{
	const Tensor<int> source(Shape{2, 2, 2});

	try {
		static_cast<void>(to_vector(source));
		FAIL() << "Expected Exceptions::ShapeError";
	} catch (const Exceptions::ShapeError& error) {
		EXPECT_STREQ(error.what(), "Array cannot be converted to a Vector.");
	}
}

TEST(ToMatrix, RankTwoMatrix)
{
	const Matrix<int> source{{1, 2, 3}, {4, 5, 6}};
	const auto result = to_matrix(source);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Matrix<int>>);
	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, 2, 3, 4, 5, 6}));
}

TEST(ToMatrix, RemovesSingletonDimensions)
{
	Tensor<int> source(Shape{1, 2, 1, 3, 1});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i + 1);
	}

	const auto result = to_matrix(source);

	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, 2, 3, 4, 5, 6}));
}

TEST(ToMatrix, PreservesNonSingletonDimensionOrder)
{
	Tensor<int> source(Shape{1, 3, 1, 2});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i);
	}

	const auto result = to_matrix(source);

	EXPECT_EQ(result.shape(), Shape({3, 2}));
	EXPECT_EQ(result(0, 0), 0);
	EXPECT_EQ(result(1, 0), 2);
	EXPECT_EQ(result(2, 1), 5);
}

TEST(ToMatrix, AcceptsZeroSizedRankTwoShape)
{
	const Matrix<int> source(0, 4);
	const auto result = to_matrix(source);

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({0, 4}));
}

TEST(ToMatrix, ReturnsIndependentStorage)
{
	Matrix<int> source{{1, 2}, {3, 4}};
	auto result = to_matrix(source);

	EXPECT_NE(result.data(), source.data());
	result(0, 0) = 99;
	EXPECT_EQ(source(0, 0), 1);
}

TEST(ToMatrix, RejectsOneNonSingletonDimension)
{
	const Tensor<int> source(Shape{1, 4, 1});

	EXPECT_THROW(static_cast<void>(to_matrix(source)), Exceptions::ShapeError);
}

TEST(ToMatrix, RejectsThreeNonSingletonDimensions)
{
	const Tensor<int> source(Shape{2, 3, 4});

	EXPECT_THROW(static_cast<void>(to_matrix(source)), Exceptions::ShapeError);
}

TEST(ToMatrix, ShapeErrorMessage)
{
	const Vector<int> source{1, 2, 3};

	try {
		static_cast<void>(to_matrix(source));
		FAIL() << "Expected Exceptions::ShapeError";
	} catch (const Exceptions::ShapeError& error) {
		EXPECT_STREQ(error.what(), "Array cannot be converted to a Matrix.");
	}
}

TEST(ToTensor, ConvertsVectorAndPreservesShape)
{
	const Vector<int> source{1, 2, 3};
	const auto result = to_tensor(source);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Tensor<int>>);
	EXPECT_EQ(result.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, 2, 3}));
}

TEST(ToTensor, ConvertsMatrixAndPreservesOrder)
{
	const Matrix<int> source{{1, 2, 3}, {4, 5, 6}};
	const auto result = to_tensor(source);

	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(result(0, 2), 3);
	EXPECT_EQ(result(1, 0), 4);
}

TEST(ToTensor, CopiesTensorIntoIndependentStorage)
{
	Tensor<int> source(Shape{2, 2}, 7);
	auto result = to_tensor(source);

	EXPECT_EQ(result.shape(), source.shape());
	EXPECT_NE(result.data(), source.data());
	result[0] = 99;
	EXPECT_EQ(source[0], 7);
}

TEST(ToTensor, PreservesEmptyShape)
{
	const Tensor<int> source(Shape{2, 0, 3});
	const auto result = to_tensor(source);

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({2, 0, 3}));
}

TEST(Astype, VectorFloatingPointToInteger)
{
	const Vector<double> source{1.9, -2.8, 3.0};
	const auto result = astype<int>(source);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Vector<int>>);
	EXPECT_EQ(result.shape(), source.shape());
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, -2, 3}));
}

TEST(Astype, MatrixIntegerToFloatingPoint)
{
	const Matrix<int> source{{1, 2}, {3, 4}};
	const auto result = astype<double>(source);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Matrix<double>>);
	EXPECT_EQ(result.shape(), source.shape());
	EXPECT_EQ(std::vector<double>(result.begin(), result.end()),
		(std::vector<double>{1.0, 2.0, 3.0, 4.0}));
}

TEST(Astype, TensorPreservesShapeAndOrder)
{
	Tensor<int> source(Shape{2, 1, 3});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i) - 2;
	}

	const auto result = astype<long double>(source);

	static_assert(std::same_as<
		std::remove_cv_t<decltype(result)>, Tensor<long double>>);
	EXPECT_EQ(result.shape(), source.shape());
	for (std::size_t i = 0; i < source.size(); ++i) {
		EXPECT_EQ(result[i], static_cast<long double>(source[i]));
	}
}

TEST(Astype, ComplexPrecisionConversion)
{
	using Source = std::complex<float>;
	using Destination = std::complex<double>;
	const Vector<Source> source{Source(1.5F, -2.0F), Source(3.25F, 4.5F)};
	const auto result = astype<Destination>(source);

	static_assert(std::same_as<
		std::remove_cv_t<decltype(result)>, Vector<Destination>>);
	EXPECT_EQ(result[0], Destination(1.5, -2.0));
	EXPECT_EQ(result[1], Destination(3.25, 4.5));
}

TEST(Astype, SameTypeStillCopies)
{
	Matrix<int> source{{1, 2}, {3, 4}};
	auto result = astype<int>(source);

	EXPECT_NE(result.data(), source.data());
	result[0] = 99;
	EXPECT_EQ(source[0], 1);
}

TEST(Astype, PreservesEmptyContainerShapes)
{
	const Vector<int> vector;
	const Matrix<int> matrix(0, 3);
	const Tensor<int> tensor(Shape{2, 0, 4});

	EXPECT_EQ(astype<double>(vector).shape(), Shape({0}));
	EXPECT_EQ(astype<double>(matrix).shape(), Shape({0, 3}));
	EXPECT_EQ(astype<double>(tensor).shape(), Shape({2, 0, 4}));
}
