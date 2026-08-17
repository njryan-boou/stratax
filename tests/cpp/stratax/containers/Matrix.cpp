#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(MatrixConstructor, DefaultConstructor)
{
	const Matrix<int> matrix;

	EXPECT_TRUE(matrix.empty());
	EXPECT_EQ(matrix.size(), 0);
	EXPECT_EQ(matrix.rank(), 2);
	EXPECT_EQ(matrix.rows(), 0);
	EXPECT_EQ(matrix.cols(), 0);
	EXPECT_EQ(matrix.shape(), Shape({0, 0}));
}

TEST(MatrixConstructor, Dimensions)
{
	const Matrix<int> matrix(2, 3);

	EXPECT_EQ(matrix.size(), 6);
	EXPECT_EQ(matrix.rank(), 2);
	EXPECT_EQ(matrix.rows(), 2);
	EXPECT_EQ(matrix.cols(), 3);
	EXPECT_EQ(matrix.shape(), Shape({2, 3}));
	EXPECT_EQ(matrix.strides(), Strides(Shape{2, 3}));

	for (const int value : matrix) {
		EXPECT_EQ(value, 0);
	}
}

TEST(MatrixConstructor, ZeroRows)
{
	const Matrix<int> matrix(0, 3);

	EXPECT_TRUE(matrix.empty());
	EXPECT_EQ(matrix.rows(), 0);
	EXPECT_EQ(matrix.cols(), 3);
	EXPECT_EQ(matrix.shape(), Shape({0, 3}));
}

TEST(MatrixConstructor, ZeroColumns)
{
	const Matrix<int> matrix(3, 0);

	EXPECT_TRUE(matrix.empty());
	EXPECT_EQ(matrix.rows(), 3);
	EXPECT_EQ(matrix.cols(), 0);
	EXPECT_EQ(matrix.shape(), Shape({3, 0}));
}

TEST(MatrixConstructor, Shape)
{
	const Matrix<int> matrix(Shape{3, 4});

	EXPECT_EQ(matrix.size(), 12);
	EXPECT_EQ(matrix.rows(), 3);
	EXPECT_EQ(matrix.cols(), 4);
	EXPECT_EQ(matrix.strides(), Strides(Shape{3, 4}));
}

TEST(MatrixConstructor, RejectsRankZeroShape)
{
	EXPECT_THROW(static_cast<void>(Matrix<int>(Shape{})), Exceptions::ShapeError);
}

TEST(MatrixConstructor, RejectsRankOneShape)
{
	EXPECT_THROW(static_cast<void>(Matrix<int>(Shape{6})), Exceptions::ShapeError);
}

TEST(MatrixConstructor, RejectsHigherRankShape)
{
	EXPECT_THROW(static_cast<void>(Matrix<int>(Shape{1, 2, 3})), Exceptions::ShapeError);
}

TEST(MatrixConstructor, ShapeErrorMessage)
{
	try {
		static_cast<void>(Matrix<int>(Shape{6}));
		FAIL() << "Expected Exceptions::ShapeError";
	} catch (const Exceptions::ShapeError& error) {
		EXPECT_STREQ(error.what(), "Matrix shape must be rank 2.");
	}
}

TEST(MatrixConstructor, RejectsElementCountOverflow)
{
	EXPECT_THROW(
		static_cast<void>(Matrix<int>(std::numeric_limits<std::size_t>::max(), 2)),
		Exceptions::DimensionError
	);
}

TEST(MatrixConstructor, FillValue)
{
	const Matrix<int> matrix(2, 3, 7);

	EXPECT_EQ(matrix.shape(), Shape({2, 3}));
	for (const int value : matrix) {
		EXPECT_EQ(value, 7);
	}
}

TEST(MatrixConstructor, InitializerList)
{
	const Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}};

	EXPECT_EQ(matrix.shape(), Shape({2, 3}));
	EXPECT_EQ(matrix.strides(), Strides(Shape{2, 3}));
	EXPECT_EQ(std::vector<int>(matrix.begin(), matrix.end()),
		(std::vector<int>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixConstructor, EmptyInitializerList)
{
	const Matrix<int> matrix(
		std::initializer_list<std::initializer_list<int>>{});

	EXPECT_TRUE(matrix.empty());
	EXPECT_EQ(matrix.shape(), Shape({0, 0}));
}

TEST(MatrixConstructor, EmptyInitializerRows)
{
	const Matrix<int> matrix{{}, {}, {}};

	EXPECT_TRUE(matrix.empty());
	EXPECT_EQ(matrix.shape(), Shape({3, 0}));
}

TEST(MatrixConstructor, RejectsRaggedInitializer)
{
	EXPECT_THROW(
		static_cast<void>(Matrix<int>{{1, 2}, {3}}),
		Exceptions::ShapeError
	);
}

TEST(MatrixConstructor, RaggedInitializerErrorMessage)
{
	try {
		static_cast<void>(Matrix<int>{{1, 2}, {3}});
		FAIL() << "Expected Exceptions::ShapeError";
	} catch (const Exceptions::ShapeError& error) {
		EXPECT_STREQ(
			error.what(),
			"Matrix initializer rows must all have the same number of columns."
		);
	}
}

TEST(MatrixConstructor, Copy)
{
	Matrix<int> original{{1, 2}, {3, 4}};
	Matrix<int> copy(original);

	EXPECT_EQ(copy.shape(), original.shape());
	EXPECT_EQ(std::vector<int>(copy.begin(), copy.end()),
		(std::vector<int>{1, 2, 3, 4}));
	EXPECT_NE(copy.data(), original.data());

	copy(0, 0) = 10;
	EXPECT_EQ(original(0, 0), 1);
}

TEST(MatrixConstructor, Move)
{
	Matrix<int> source{{1, 2}, {3, 4}};
	int* source_data = source.data();

	Matrix<int> destination(std::move(source));

	EXPECT_EQ(destination.data(), source_data);
	EXPECT_EQ(destination.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(destination.begin(), destination.end()),
		(std::vector<int>{1, 2, 3, 4}));
	EXPECT_TRUE(source.empty());
}

TEST(MatrixAssignment, CopyAssignment)
{
	const Matrix<int> source{{1, 2}, {3, 4}};
	Matrix<int> destination(1, 3, 9);

	destination = source;

	EXPECT_EQ(destination.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(destination.begin(), destination.end()),
		(std::vector<int>{1, 2, 3, 4}));
	EXPECT_NE(destination.data(), source.data());
}

TEST(MatrixAssignment, SelfCopyAssignment)
{
	Matrix<int> matrix{{1, 2}, {3, 4}};
	int* data = matrix.data();

	matrix = matrix;

	EXPECT_EQ(matrix.data(), data);
	EXPECT_EQ(matrix.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(matrix.begin(), matrix.end()),
		(std::vector<int>{1, 2, 3, 4}));
}

TEST(MatrixAssignment, MoveAssignment)
{
	Matrix<int> source{{1, 2}, {3, 4}};
	Matrix<int> destination(1, 3, 9);
	int* source_data = source.data();

	destination = std::move(source);

	EXPECT_EQ(destination.data(), source_data);
	EXPECT_EQ(destination.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(destination.begin(), destination.end()),
		(std::vector<int>{1, 2, 3, 4}));
	EXPECT_TRUE(source.empty());
}

TEST(MatrixAssignment, SelfMoveAssignment)
{
	Matrix<int> matrix{{1, 2}, {3, 4}};
	int* data = matrix.data();

	matrix = std::move(matrix);

	EXPECT_EQ(matrix.data(), data);
	EXPECT_EQ(matrix.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(matrix.begin(), matrix.end()),
		(std::vector<int>{1, 2, 3, 4}));
}

TEST(MatrixTypes, Aliases)
{
	using TestMatrix = Matrix<int>;

	static_assert(std::same_as<TestMatrix::value_type, int>);
	static_assert(std::same_as<TestMatrix::size_type, std::size_t>);
	static_assert(std::same_as<TestMatrix::difference_type, std::ptrdiff_t>);
	static_assert(std::same_as<TestMatrix::reference, int&>);
	static_assert(std::same_as<TestMatrix::const_reference, const int&>);
	static_assert(std::same_as<TestMatrix::pointer, int*>);
	static_assert(std::same_as<TestMatrix::const_pointer, const int*>);
	static_assert(std::contiguous_iterator<TestMatrix::iterator>);
	static_assert(std::contiguous_iterator<TestMatrix::const_iterator>);
	static_assert(std::same_as<
		TestMatrix::reverse_iterator,
		std::reverse_iterator<TestMatrix::iterator>
	>);
	static_assert(std::same_as<
		TestMatrix::const_reverse_iterator,
		std::reverse_iterator<TestMatrix::const_iterator>
	>);
}

TEST(MatrixMeta, RowsAndColumns)
{
	const Matrix<int> matrix(2, 3);

	EXPECT_EQ(matrix.rows(), 2);
	EXPECT_EQ(matrix.cols(), 3);
	static_assert(std::same_as<decltype(matrix.rows()), std::size_t>);
	static_assert(std::same_as<decltype(matrix.cols()), std::size_t>);
	static_assert(noexcept(matrix.rows()));
	static_assert(noexcept(matrix.cols()));
}

TEST(MatrixAccess, MutableAndConstTwoDimensionalAccess)
{
	Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}};
	const Matrix<int>& const_matrix = matrix;

	matrix(1, 2) = 60;

	EXPECT_EQ(matrix(1, 2), 60);
	EXPECT_EQ(const_matrix(1, 2), 60);
	EXPECT_EQ(matrix[5], 60);
	static_assert(std::same_as<decltype(matrix(0, 0)), int&>);
	static_assert(std::same_as<decltype(const_matrix(0, 0)), const int&>);
}

TEST(MatrixAccess, RowMajorLayout)
{
	const Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}};

	EXPECT_EQ(matrix(0, 0), 1);
	EXPECT_EQ(matrix(0, 2), 3);
	EXPECT_EQ(matrix(1, 0), 4);
	EXPECT_EQ(matrix(1, 2), 6);
	EXPECT_EQ(std::vector<int>(matrix.begin(), matrix.end()),
		(std::vector<int>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixAccess, CheckedPositiveIndices)
{
	Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}};
	const Matrix<int>& const_matrix = matrix;

	matrix.at(1, 2) = 60;

	EXPECT_EQ(matrix.at(0, 0), 1);
	EXPECT_EQ(matrix.at(1, 2), 60);
	EXPECT_EQ(const_matrix.at(1, 1), 5);
}

TEST(MatrixAccess, CheckedNegativeIndices)
{
	const Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}};

	EXPECT_EQ(matrix.at(-1, -1), 6);
	EXPECT_EQ(matrix.at(-1, 0), 4);
	EXPECT_EQ(matrix.at(0, -2), 2);
	EXPECT_EQ(matrix.at(-2, -3), 1);
}

TEST(MatrixAccess, CheckedTwoDimensionalIndexOutOfRange)
{
	const Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}};

	EXPECT_THROW(static_cast<void>(matrix.at(2, 0)), Exceptions::IndexError);
	EXPECT_THROW(static_cast<void>(matrix.at(-3, 0)), Exceptions::IndexError);
	EXPECT_THROW(static_cast<void>(matrix.at(0, 3)), Exceptions::IndexError);
	EXPECT_THROW(static_cast<void>(matrix.at(0, -4)), Exceptions::IndexError);
}

TEST(MatrixAccess, InheritedFlatAt)
{
	Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}};
	const Matrix<int>& const_matrix = matrix;

	matrix.at(1) = 20;

	EXPECT_EQ(matrix.at(1), 20);
	EXPECT_EQ(const_matrix.at(-1), 6);
	EXPECT_THROW(static_cast<void>(matrix.at(6)), Exceptions::IndexError);
}

TEST(MatrixIterator, ForwardAndReverseTraversal)
{
	Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}};

	EXPECT_EQ(std::vector<int>(matrix.begin(), matrix.end()),
		(std::vector<int>{1, 2, 3, 4, 5, 6}));
	EXPECT_EQ(std::vector<int>(matrix.crbegin(), matrix.crend()),
		(std::vector<int>{6, 5, 4, 3, 2, 1}));

	for (int& value : matrix) {
		value *= 2;
	}
	EXPECT_EQ(std::vector<int>(matrix.cbegin(), matrix.cend()),
		(std::vector<int>{2, 4, 6, 8, 10, 12}));
}

TEST(MatrixModifier, Fill)
{
	Matrix<int> matrix{{1, 2}, {3, 4}};

	matrix.fill(9);

	EXPECT_EQ(std::vector<int>(matrix.begin(), matrix.end()),
		(std::vector<int>{9, 9, 9, 9}));
}

TEST(MatrixModifier, MemberSwap)
{
	Matrix<int> lhs{{1, 2, 3}, {4, 5, 6}};
	Matrix<int> rhs{{7, 8}};
	int* lhs_data = lhs.data();
	int* rhs_data = rhs.data();

	lhs.swap(rhs);

	EXPECT_EQ(lhs.data(), rhs_data);
	EXPECT_EQ(rhs.data(), lhs_data);
	EXPECT_EQ(lhs.shape(), Shape({1, 2}));
	EXPECT_EQ(rhs.shape(), Shape({2, 3}));
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()), (std::vector<int>{7, 8}));
	EXPECT_EQ(std::vector<int>(rhs.begin(), rhs.end()),
		(std::vector<int>{1, 2, 3, 4, 5, 6}));
}

TEST(MatrixModifier, NonMemberSwap)
{
	Matrix<int> lhs{{1, 2}, {3, 4}};
	Matrix<int> rhs{{5, 6, 7}};

	using std::swap;
	swap(lhs, rhs);

	EXPECT_EQ(lhs.shape(), Shape({1, 3}));
	EXPECT_EQ(rhs.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()),
		(std::vector<int>{5, 6, 7}));
	EXPECT_EQ(std::vector<int>(rhs.begin(), rhs.end()),
		(std::vector<int>{1, 2, 3, 4}));
}

TEST(MatrixModifier, SelfSwap)
{
	Matrix<int> matrix{{1, 2}, {3, 4}};
	int* data = matrix.data();

	matrix.swap(matrix);

	EXPECT_EQ(matrix.data(), data);
	EXPECT_EQ(matrix.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(matrix.begin(), matrix.end()),
		(std::vector<int>{1, 2, 3, 4}));
}

TEST(MatrixTraits, NumericTypesAreSupported)
{
	static_assert(std::default_initializable<Matrix<int>>);
	static_assert(std::default_initializable<Matrix<float>>);
	static_assert(std::default_initializable<Matrix<double>>);
	static_assert(std::is_copy_constructible_v<Matrix<int>>);
	static_assert(std::is_move_constructible_v<Matrix<int>>);
	static_assert(std::is_copy_assignable_v<Matrix<int>>);
	static_assert(std::is_move_assignable_v<Matrix<int>>);
	static_assert(std::is_nothrow_swappable_v<Matrix<int>>);
}
