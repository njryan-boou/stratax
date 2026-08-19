#include <gtest/gtest.h>

#include <concepts>
#include <type_traits>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;
using stratax::indexing::slice;

namespace {

template<typename T>
std::vector<T> view_values(const ArrayView<T>& view)
{
	std::vector<T> values;
	values.reserve(view.size());

	for (std::size_t i = 0; i < view.size(); ++i)
	{
		values.push_back(view(i));
	}

	return values;
}

} // namespace

TEST(SlicingTypes, Aliases)
{
	static_assert(std::same_as<stratax::indexing::size_type, std::size_t>);
	static_assert(std::same_as<stratax::indexing::difference_type, std::ptrdiff_t>);
}

TEST(VectorSlicing, FullRange)
{
	Vector<int> source{0, 1, 2, 3, 4};
	const auto result = slice(source, Slice{0, 5});

	static_assert(std::same_as<
		std::remove_cv_t<decltype(result)>,
		ArrayView<int>>);
	EXPECT_EQ(result.shape(), Shape({5}));
	EXPECT_EQ(result.strides(), Shape({1}));
	EXPECT_EQ(result.data(), source.data());
	EXPECT_EQ(view_values(result), (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(VectorSlicing, PositiveStep)
{
	Vector<int> source{0, 1, 2, 3, 4, 5, 6, 7};
	const auto result = slice(source, Slice{1, 8, 3});

	EXPECT_EQ(result.shape(), Shape({3}));
	EXPECT_EQ(result.strides(), Shape({3}));
	EXPECT_EQ(result.data(), source.data() + 1);
	EXPECT_EQ(view_values(result), (std::vector<int>{1, 4, 7}));
}

TEST(VectorSlicing, NegativeBounds)
{
	Vector<int> source{0, 1, 2, 3, 4, 5};
	const auto result = slice(source, Slice{-5, -1, 2});

	EXPECT_EQ(result.data(), source.data() + 1);
	EXPECT_EQ(result.strides(), Shape({2}));
	EXPECT_EQ(view_values(result), (std::vector<int>{1, 3}));
}

TEST(VectorSlicing, ReverseFullRange)
{
	Vector<int> source{0, 1, 2, 3, 4};

	EXPECT_THROW(
		static_cast<void>(slice(source, Slice{-1, -1, -1})),
		Exceptions::IndexError);
}

TEST(VectorSlicing, ReverseNegativeBounds)
{
	Vector<int> source{0, 1, 2, 3, 4, 5};

	EXPECT_THROW(
		static_cast<void>(slice(source, Slice{-1, -5, -2})),
		Exceptions::IndexError);
}

TEST(VectorSlicing, ClampsBounds)
{
	Vector<int> source{0, 1, 2, 3, 4};
	const auto forward = slice(source, Slice{-100, 100});

	EXPECT_EQ(view_values(forward), (std::vector<int>{0, 1, 2, 3, 4}));
	EXPECT_THROW(
		static_cast<void>(slice(source, Slice{100, -100, -1})),
		Exceptions::IndexError);
}

TEST(VectorSlicing, EmptyRanges)
{
	Vector<int> source{0, 1, 2, 3, 4};
	const auto forward = slice(source, Slice{4, 1, 1});

	EXPECT_EQ(forward.size(), 0);
	EXPECT_EQ(forward.shape(), Shape({0}));
	EXPECT_THROW(
		static_cast<void>(slice(source, Slice{1, 4, -1})),
		Exceptions::IndexError);
}

TEST(VectorSlicing, EmptySource)
{
	Vector<int> source;
	const auto forward = slice(source, Slice{0, 10});

	EXPECT_EQ(forward.size(), 0);
	EXPECT_EQ(forward.shape(), Shape({0}));
	EXPECT_THROW(
		static_cast<void>(slice(source, Slice{-1, -1, -1})),
		Exceptions::IndexError);
}

TEST(VectorSlicing, SharesSourceStorage)
{
	Vector<int> source{0, 1, 2, 3, 4};
	auto result = slice(source, Slice{1, 4});

	result(0) = 99;

	EXPECT_EQ(source[1], 99);
	EXPECT_EQ(result.data(), source.data() + 1);
}

TEST(MatrixSlicing, FullRange)
{
	const Matrix<int> source{
		{0, 1, 2},
		{3, 4, 5}
	};
	const auto result = slice(source, Slice{0, 2}, Slice{0, 3});

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Matrix<int>>);
	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST(MatrixSlicing, StridedRowsAndColumns)
{
	const Matrix<int> source{
		{0, 1, 2, 3, 4},
		{5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14},
		{15, 16, 17, 18, 19}
	};
	const auto result = slice(source, Slice{1, 4, 2}, Slice{0, 5, 2});

	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{5, 7, 9, 15, 17, 19}));
}

TEST(MatrixSlicing, NegativeBounds)
{
	const Matrix<int> source{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 10, 11}
	};
	const auto result = slice(source, Slice{-2, 3}, Slice{-3, -1});

	EXPECT_EQ(result.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{5, 6, 9, 10}));
}

TEST(MatrixSlicing, ReversesBothDimensions)
{
	const Matrix<int> source{
		{0, 1, 2},
		{3, 4, 5}
	};
	const auto result = slice(
		source,
		Slice{-1, -1, -1},
		Slice{-1, -1, -1});

	EXPECT_EQ(result.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{5, 4, 3, 2, 1, 0}));
}

TEST(MatrixSlicing, EmptyRowSelection)
{
	const Matrix<int> source(3, 4, 1);
	const auto result = slice(source, Slice{2, 1}, Slice{0, 4});

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({0, 4}));
}

TEST(MatrixSlicing, EmptyColumnSelection)
{
	const Matrix<int> source(3, 4, 1);
	const auto result = slice(source, Slice{0, 3}, Slice{3, 1});

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({3, 0}));
}

TEST(MatrixSlicing, EmptySource)
{
	const Matrix<int> source;
	const auto result = slice(source, Slice{0, 1}, Slice{0, 1});

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({0, 0}));
}

TEST(MatrixSlicing, ReturnsIndependentStorage)
{
	Matrix<int> source{{0, 1}, {2, 3}};
	auto result = slice(source, Slice{0, 2}, Slice{0, 2});

	result(0, 0) = 99;

	EXPECT_EQ(source(0, 0), 0);
	EXPECT_NE(result.data(), source.data());
}

TEST(TensorVariadicSlicing, FullRange)
{
	Tensor<int> source(Shape{2, 3, 4});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i);
	}
	const auto result = slice(
		source,
		Slice{0, 2},
		Slice{0, 3},
		Slice{0, 4});

	static_assert(std::same_as<std::remove_cv_t<decltype(result)>, Tensor<int>>);
	EXPECT_EQ(result.shape(), Shape({2, 3, 4}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		std::vector<int>(source.begin(), source.end()));
}

TEST(TensorVariadicSlicing, StridedDimensions)
{
	Tensor<int> source(Shape{2, 3, 4});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i);
	}
	const auto result = slice(
		source,
		Slice{0, 2},
		Slice{0, 3, 2},
		Slice{1, 4, 2});

	EXPECT_EQ(result.shape(), Shape({2, 2, 2}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{1, 3, 9, 11, 13, 15, 21, 23}));
}

TEST(TensorVariadicSlicing, ReversesEveryDimension)
{
	Tensor<int> source(Shape{2, 3, 4});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i);
	}
	const auto result = slice(
		source,
		Slice{-1, -1, -1},
		Slice{-1, -1, -1},
		Slice{-1, -1, -1});

	std::vector<int> expected;
	for (int value = 23; value >= 0; --value) {
		expected.push_back(value);
	}

	EXPECT_EQ(result.shape(), Shape({2, 3, 4}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()), expected);
}

TEST(TensorVariadicSlicing, EmptyDimension)
{
	const Tensor<int> source(Shape{2, 3, 4}, 1);
	const auto result = slice(
		source,
		Slice{0, 2},
		Slice{2, 1},
		Slice{0, 4});

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({2, 0, 4}));
}

TEST(TensorVariadicSlicing, RankZeroTensor)
{
	const Tensor<int> source(Shape{});
	const auto result = slice(source);

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.rank(), 0);
	EXPECT_EQ(result.shape(), Shape{});
}

TEST(TensorVariadicSlicing, RejectsRankMismatch)
{
	const Tensor<int> source(Shape{2, 3, 4});

	EXPECT_THROW(
		static_cast<void>(slice(source, Slice{0, 2}, Slice{0, 3})),
		Exceptions::IndexError
	);
}

TEST(TensorVariadicSlicing, RankMismatchErrorMessage)
{
	const Tensor<int> source(Shape{2, 3, 4});

	try {
		static_cast<void>(slice(source, Slice{0, 2}, Slice{0, 3}));
		FAIL() << "Expected Exceptions::IndexError";
	} catch (const Exceptions::IndexError& error) {
		EXPECT_STREQ(error.what(), "Tensor slice rank must match tensor rank.");
	}
}

TEST(TensorVariadicSlicing, ReturnsIndependentStorage)
{
	Tensor<int> source(Shape{2, 2}, 1);
	auto result = slice(source, Slice{0, 2}, Slice{0, 2});

	result(0, 0) = 99;

	EXPECT_EQ(source(0, 0), 1);
	EXPECT_NE(result.data(), source.data());
}

TEST(TensorVectorSlicing, FullRange)
{
	Tensor<int> source(Shape{2, 3, 4});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i);
	}
	const std::vector<Slice> slices{
		Slice{0, 2},
		Slice{0, 3},
		Slice{0, 4}
	};
	const auto result = slice(source, slices);

	EXPECT_EQ(result.shape(), source.shape());
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		std::vector<int>(source.begin(), source.end()));
}

TEST(TensorVectorSlicing, StridedAndNegativeDimensions)
{
	Tensor<int> source(Shape{2, 3, 4});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i);
	}
	const std::vector<Slice> slices{
		Slice{-1, -1, -1},
		Slice{0, 3, 2},
		Slice{1, 4, 2}
	};
	const auto result = slice(source, slices);

	EXPECT_EQ(result.shape(), Shape({2, 2, 2}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{13, 15, 21, 23, 1, 3, 9, 11}));
}

TEST(TensorVectorSlicing, EmptyDimension)
{
	const Tensor<int> source(Shape{2, 3, 4}, 1);
	const std::vector<Slice> slices{
		Slice{0, 2},
		Slice{2, 1},
		Slice{0, 4}
	};
	const auto result = slice(source, slices);

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({2, 0, 4}));
}

TEST(TensorVectorSlicing, RankZeroTensor)
{
	const Tensor<int> source(Shape{});
	const std::vector<Slice> slices;
	const auto result = slice(source, slices);

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.rank(), 0);
	EXPECT_EQ(result.shape(), Shape{});
}

TEST(TensorVectorSlicing, RejectsRankMismatch)
{
	const Tensor<int> source(Shape{2, 3, 4});
	const std::vector<Slice> slices{Slice{0, 2}, Slice{0, 3}};

	EXPECT_THROW(static_cast<void>(slice(source, slices)), Exceptions::DimensionError);
}

TEST(TensorVectorSlicing, RankMismatchErrorMessage)
{
	const Tensor<int> source(Shape{2, 3, 4});
	const std::vector<Slice> slices{Slice{0, 2}, Slice{0, 3}};

	try {
		static_cast<void>(slice(source, slices));
		FAIL() << "Expected Exceptions::DimensionError";
	} catch (const Exceptions::DimensionError& error) {
		EXPECT_STREQ(error.what(), "Slice rank must match tensor rank.");
	}
}

TEST(TensorVectorSlicing, MatchesVariadicOverload)
{
	Tensor<int> source(Shape{2, 3, 4});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i);
	}
	const Slice first{-1, -1, -1};
	const Slice second{0, 3, 2};
	const Slice third{1, 4, 2};

	const auto variadic = slice(source, first, second, third);
	const auto vector_based = slice(
		source,
		std::vector<Slice>{first, second, third});

	EXPECT_EQ(vector_based.shape(), variadic.shape());
	EXPECT_EQ(
		std::vector<int>(vector_based.begin(), vector_based.end()),
		std::vector<int>(variadic.begin(), variadic.end())
	);
}
