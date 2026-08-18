#include <gtest/gtest.h>

#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;
using namespace stratax::manipulation;

TEST(Reshape, VectorToMatrixShapedTensor)
{
	const Vector<int> source{0, 1, 2, 3, 4, 5};
	const auto result = reshape(source, Shape{2, 3});

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Tensor<int>>);
	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(result.strides(), Strides(Shape{2, 3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST(Reshape, MatrixToRankOneTensor)
{
	const Matrix<int> source{{0, 1, 2}, {3, 4, 5}};
	const auto result = reshape(source, Shape{6});

	EXPECT_EQ(result.shape(), Shape({6}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST(Reshape, MatrixToHigherRankTensor)
{
	const Matrix<int> source{{0, 1, 2}, {3, 4, 5}};
	const auto result = reshape(source, Shape{1, 2, 3, 1});

	EXPECT_EQ(result.shape(), Shape({1, 2, 3, 1}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST(Reshape, TensorToDifferentShape)
{
	Tensor<int> source(Shape{2, 3, 4});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i);
	}
	const auto result = reshape(source, Shape{4, 3, 2});

	EXPECT_EQ(result.shape(), Shape({4, 3, 2}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		std::vector<int>(source.begin(), source.end()));
}

TEST(Reshape, SameShapeStillCopies)
{
	Tensor<int> source(Shape{2, 2}, 1);
	auto result = reshape(source, source.shape());

	EXPECT_EQ(result.shape(), source.shape());
	EXPECT_NE(result.data(), source.data());

	result[0] = 99;
	EXPECT_EQ(source[0], 1);
}

TEST(Reshape, PreservesFlatRowMajorOrder)
{
	const Matrix<int> source{{1, 2, 3}, {4, 5, 6}};
	const auto result = reshape(source, Shape{3, 2});

	EXPECT_EQ(result(0, 0), 1);
	EXPECT_EQ(result(0, 1), 2);
	EXPECT_EQ(result(1, 0), 3);
	EXPECT_EQ(result(1, 1), 4);
	EXPECT_EQ(result(2, 0), 5);
	EXPECT_EQ(result(2, 1), 6);
}

TEST(Reshape, RejectsSmallerTarget)
{
	const Vector<int> source{0, 1, 2, 3, 4, 5};

	EXPECT_THROW(
		static_cast<void>(reshape(source, Shape{5})),
		Exceptions::ShapeError
	);
}

TEST(Reshape, RejectsLargerTarget)
{
	const Vector<int> source{0, 1, 2, 3, 4, 5};

	EXPECT_THROW(
		static_cast<void>(reshape(source, Shape{7})),
		Exceptions::ShapeError
	);
}

TEST(Reshape, SizeMismatchErrorMessage)
{
	const Vector<int> source{0, 1, 2};

	try {
		static_cast<void>(reshape(source, Shape{2, 2}));
		FAIL() << "Expected Exceptions::ShapeError";
	} catch (const Exceptions::ShapeError& error) {
		EXPECT_STREQ(
			error.what(),
			"Reshape size must match original array size."
		);
	}
}

TEST(Reshape, TargetElementCountOverflow)
{
	const Vector<int> source{1};

	EXPECT_THROW(
		static_cast<void>(reshape(
			source,
			Shape{std::numeric_limits<std::size_t>::max(), 2})),
		Exceptions::DimensionError
	);
}

TEST(Reshape, EmptyVectorToRankZeroTensor)
{
	const Vector<int> source;
	const auto result = reshape(source, Shape{});

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.rank(), 0);
	EXPECT_EQ(result.shape(), Shape{});
}

TEST(Reshape, EmptyVectorToZeroDimensionTensor)
{
	const Vector<int> source;
	const auto result = reshape(source, Shape{2, 0, 4});

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({2, 0, 4}));
}

TEST(Reshape, ZeroDimensionTensorToAnotherEmptyShape)
{
	const Tensor<int> source(Shape{2, 0, 4});
	const auto result = reshape(source, Shape{0, 7});

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({0, 7}));
}

TEST(Reshape, PreservesComplexValueType)
{
	using Complex = std::complex<double>;
	const Vector<Complex> source{
		Complex(1.0, 2.0),
		Complex(3.0, 4.0),
		Complex(5.0, 6.0),
		Complex(7.0, 8.0)
	};
	const auto result = reshape(source, Shape{2, 2});

	static_assert(std::same_as<
		typename std::remove_cv_t<decltype(result)>::value_type,
		Complex
	>);
	EXPECT_EQ(std::vector<Complex>(result.begin(), result.end()),
		std::vector<Complex>(source.begin(), source.end()));
}

TEST(Flatten, Vector)
{
	const Vector<int> source{1, 2, 3, 4};
	const auto result = flatten(source);

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Vector<int>>);
	EXPECT_EQ(result.shape(), Shape({4}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, 2, 3, 4}));
}

TEST(Flatten, Matrix)
{
	const Matrix<int> source{{1, 2, 3}, {4, 5, 6}};
	const auto result = flatten(source);

	EXPECT_EQ(result.shape(), Shape({6}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, 2, 3, 4, 5, 6}));
}

TEST(Flatten, Tensor)
{
	Tensor<int> source(Shape{2, 3, 2});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i);
	}
	const auto result = flatten(source);

	EXPECT_EQ(result.shape(), Shape({12}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		std::vector<int>(source.begin(), source.end()));
}

TEST(Flatten, ReturnsIndependentStorage)
{
	Matrix<int> source{{1, 2}, {3, 4}};
	auto result = flatten(source);

	EXPECT_NE(result.data(), source.data());
	result[0] = 99;
	EXPECT_EQ(source[0], 1);
}

TEST(Flatten, EmptyContainers)
{
	const Vector<int> vector;
	const Matrix<int> matrix;
	const Tensor<int> tensor(Shape{});

	const auto flat_vector = flatten(vector);
	const auto flat_matrix = flatten(matrix);
	const auto flat_tensor = flatten(tensor);

	EXPECT_TRUE(flat_vector.empty());
	EXPECT_EQ(flat_vector.shape(), Shape({0}));
	EXPECT_TRUE(flat_matrix.empty());
	EXPECT_EQ(flat_matrix.shape(), Shape({0}));
	EXPECT_TRUE(flat_tensor.empty());
	EXPECT_EQ(flat_tensor.shape(), Shape({0}));
}

TEST(Flatten, PreservesComplexValueType)
{
	using Complex = std::complex<float>;
	const Matrix<Complex> source{
		{Complex(1.0F, 2.0F), Complex(3.0F, 4.0F)},
		{Complex(5.0F, 6.0F), Complex(7.0F, 8.0F)}
	};
	const auto result = flatten(source);

	static_assert(std::same_as<
		typename std::remove_cv_t<decltype(result)>::value_type,
		Complex
	>);
	EXPECT_EQ(std::vector<Complex>(result.begin(), result.end()),
		std::vector<Complex>(source.begin(), source.end()));
}
