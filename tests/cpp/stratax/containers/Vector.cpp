#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(VectorConstructor, DefaultConstructor)
{
	const Vector<int> vector;

	EXPECT_TRUE(vector.empty());
	EXPECT_EQ(vector.size(), 0);
	EXPECT_EQ(vector.rank(), 1);
	EXPECT_EQ(vector.shape(), Shape({0}));
	EXPECT_EQ(vector.strides(), Strides({1}));
}

TEST(VectorConstructor, Size)
{
	const Vector<int> vector(4);

	EXPECT_EQ(vector.size(), 4);
	EXPECT_EQ(vector.rank(), 1);
	EXPECT_EQ(vector.shape(), Shape({4}));
	EXPECT_EQ(vector.strides(), Strides({1}));

	for (const int value : vector) {
		EXPECT_EQ(value, 0);
	}
}

TEST(VectorConstructor, ZeroSize)
{
	const Vector<int> vector(0);

	EXPECT_TRUE(vector.empty());
	EXPECT_EQ(vector.rank(), 1);
	EXPECT_EQ(vector.shape(), Shape({0}));
}

TEST(VectorConstructor, Shape)
{
	const Vector<int> vector(Shape{5});

	EXPECT_EQ(vector.size(), 5);
	EXPECT_EQ(vector.shape(), Shape({5}));
	EXPECT_EQ(vector.strides(), Strides({1}));
}

TEST(VectorConstructor, RejectsRankZeroShape)
{
	EXPECT_THROW(static_cast<void>(Vector<int>(Shape{})), Exceptions::ShapeError);
}

TEST(VectorConstructor, RejectsHigherRankShape)
{
	EXPECT_THROW(static_cast<void>(Vector<int>(Shape{2, 3})), Exceptions::ShapeError);
}

TEST(VectorConstructor, ShapeErrorMessage)
{
	try {
		static_cast<void>(Vector<int>(Shape{2, 3}));
		FAIL() << "Expected Exceptions::ShapeError";
	} catch (const Exceptions::ShapeError& error) {
		EXPECT_STREQ(error.what(), "Vector shape must be rank 1.");
	}
}

TEST(VectorConstructor, FillValue)
{
	const Vector<int> vector(4, 7);

	EXPECT_EQ(vector.size(), 4);
	EXPECT_EQ(vector.shape(), Shape({4}));
	for (const int value : vector) {
		EXPECT_EQ(value, 7);
	}
}

TEST(VectorConstructor, InitializerList)
{
	const Vector<int> vector{1, 2, 3, 4};

	EXPECT_EQ(vector.size(), 4);
	EXPECT_EQ(vector.shape(), Shape({4}));
	EXPECT_EQ(vector[0], 1);
	EXPECT_EQ(vector[1], 2);
	EXPECT_EQ(vector[2], 3);
	EXPECT_EQ(vector[3], 4);
}

TEST(VectorConstructor, EmptyInitializerList)
{
	const Vector<int> vector(std::initializer_list<int>{});

	EXPECT_TRUE(vector.empty());
	EXPECT_EQ(vector.rank(), 1);
	EXPECT_EQ(vector.shape(), Shape({0}));
}

TEST(VectorConstructor, Copy)
{
	Vector<int> original{1, 2, 3};
	Vector<int> copy(original);

	EXPECT_EQ(copy.shape(), original.shape());
	EXPECT_EQ(std::vector<int>(copy.begin(), copy.end()), (std::vector<int>{1, 2, 3}));
	EXPECT_NE(copy.data(), original.data());

	copy[0] = 10;
	EXPECT_EQ(original[0], 1);
}

TEST(VectorConstructor, Move)
{
	Vector<int> source{1, 2, 3};
	int* source_data = source.data();

	Vector<int> destination(std::move(source));

	EXPECT_EQ(destination.data(), source_data);
	EXPECT_EQ(destination.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(destination.begin(), destination.end()), (std::vector<int>{1, 2, 3}));
	EXPECT_TRUE(source.empty());
}

TEST(VectorAssignment, CopyAssignment)
{
	const Vector<int> source{1, 2, 3};
	Vector<int> destination{4, 5};

	destination = source;

	EXPECT_EQ(destination.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(destination.begin(), destination.end()), (std::vector<int>{1, 2, 3}));
	EXPECT_NE(destination.data(), source.data());
}

TEST(VectorAssignment, SelfCopyAssignment)
{
	Vector<int> vector{1, 2, 3};
	int* data = vector.data();

	vector = vector;

	EXPECT_EQ(vector.data(), data);
	EXPECT_EQ(std::vector<int>(vector.begin(), vector.end()), (std::vector<int>{1, 2, 3}));
}

TEST(VectorAssignment, MoveAssignment)
{
	Vector<int> source{1, 2, 3};
	Vector<int> destination{4, 5};
	int* source_data = source.data();

	destination = std::move(source);

	EXPECT_EQ(destination.data(), source_data);
	EXPECT_EQ(destination.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(destination.begin(), destination.end()), (std::vector<int>{1, 2, 3}));
	EXPECT_TRUE(source.empty());
}

TEST(VectorAssignment, SelfMoveAssignment)
{
	Vector<int> vector{1, 2, 3};
	int* data = vector.data();

	vector = std::move(vector);

	EXPECT_EQ(vector.data(), data);
	EXPECT_EQ(vector.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(vector.begin(), vector.end()), (std::vector<int>{1, 2, 3}));
}

TEST(VectorTypes, Aliases)
{
	using TestVector = Vector<int>;

	static_assert(std::same_as<TestVector::value_type, int>);
	static_assert(std::same_as<TestVector::size_type, std::size_t>);
	static_assert(std::same_as<TestVector::difference_type, std::ptrdiff_t>);
	static_assert(std::same_as<TestVector::reference, int&>);
	static_assert(std::same_as<TestVector::const_reference, const int&>);
	static_assert(std::same_as<TestVector::pointer, int*>);
	static_assert(std::same_as<TestVector::const_pointer, const int*>);
	static_assert(std::contiguous_iterator<TestVector::iterator>);
	static_assert(std::contiguous_iterator<TestVector::const_iterator>);
	static_assert(std::same_as<TestVector::reverse_iterator, std::reverse_iterator<TestVector::iterator>>);
	static_assert(std::same_as<TestVector::const_reverse_iterator, std::reverse_iterator<TestVector::const_iterator>>);
}

TEST(VectorAccess, MutableAndConstSubscript)
{
	Vector<int> vector{1, 2, 3};
	const Vector<int>& const_vector = vector;

	vector[1] = 20;

	EXPECT_EQ(vector[1], 20);
	EXPECT_EQ(const_vector[1], 20);
	static_assert(std::same_as<decltype(vector[0]), int&>);
	static_assert(std::same_as<decltype(const_vector[0]), const int&>);
}

TEST(VectorAccess, CheckedPositiveAndNegativeIndices)
{
	Vector<int> vector{10, 20, 30};
	const Vector<int>& const_vector = vector;

	EXPECT_EQ(vector.at(0), 10);
	EXPECT_EQ(vector.at(-1), 30);
	EXPECT_EQ(const_vector.at(-2), 20);
	vector.at(-1) = 40;
	EXPECT_EQ(vector.back(), 40);
}

TEST(VectorAccess, CheckedIndexOutOfRange)
{
	const Vector<int> vector{1, 2, 3};

	EXPECT_THROW(static_cast<void>(vector.at(3)), Exceptions::IndexError);
	EXPECT_THROW(static_cast<void>(vector.at(-4)), Exceptions::IndexError);
}

TEST(VectorAccess, DataFrontAndBack)
{
	Vector<int> vector{1, 2, 3};
	const Vector<int>& const_vector = vector;

	EXPECT_EQ(vector.data(), &vector[0]);
	EXPECT_EQ(const_vector.data(), &const_vector[0]);
	EXPECT_EQ(vector.front(), 1);
	EXPECT_EQ(vector.back(), 3);

	vector.front() = 10;
	vector.back() = 30;
	EXPECT_EQ(std::vector<int>(vector.begin(), vector.end()), (std::vector<int>{10, 2, 30}));
}

TEST(VectorIterator, ForwardTraversal)
{
	Vector<int> vector{1, 2, 3, 4};

	EXPECT_EQ(std::vector<int>(vector.begin(), vector.end()), (std::vector<int>{1, 2, 3, 4}));
	EXPECT_EQ(std::vector<int>(vector.cbegin(), vector.cend()), (std::vector<int>{1, 2, 3, 4}));

	for (int& value : vector) {
		value *= 2;
	}
	EXPECT_EQ(std::vector<int>(vector.begin(), vector.end()), (std::vector<int>{2, 4, 6, 8}));
}

TEST(VectorIterator, ReverseTraversal)
{
	const Vector<int> vector{1, 2, 3, 4};

	EXPECT_EQ(std::vector<int>(vector.rbegin(), vector.rend()), (std::vector<int>{4, 3, 2, 1}));
	EXPECT_EQ(std::vector<int>(vector.crbegin(), vector.crend()), (std::vector<int>{4, 3, 2, 1}));
}

TEST(VectorModifier, Fill)
{
	Vector<int> vector{1, 2, 3};

	vector.fill(9);

	EXPECT_EQ(std::vector<int>(vector.begin(), vector.end()), (std::vector<int>{9, 9, 9}));
}

TEST(VectorModifier, MemberSwap)
{
	Vector<int> lhs{1, 2, 3};
	Vector<int> rhs{4, 5};
	int* lhs_data = lhs.data();
	int* rhs_data = rhs.data();

	lhs.swap(rhs);

	EXPECT_EQ(lhs.data(), rhs_data);
	EXPECT_EQ(rhs.data(), lhs_data);
	EXPECT_EQ(lhs.shape(), Shape({2}));
	EXPECT_EQ(rhs.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()), (std::vector<int>{4, 5}));
	EXPECT_EQ(std::vector<int>(rhs.begin(), rhs.end()), (std::vector<int>{1, 2, 3}));
}

TEST(VectorModifier, NonMemberSwap)
{
	Vector<int> lhs{1, 2, 3};
	Vector<int> rhs{4};

	using std::swap;
	swap(lhs, rhs);

	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()), (std::vector<int>{4}));
	EXPECT_EQ(std::vector<int>(rhs.begin(), rhs.end()), (std::vector<int>{1, 2, 3}));
}

TEST(VectorModifier, SelfSwap)
{
	Vector<int> vector{1, 2, 3};
	int* data = vector.data();

	vector.swap(vector);

	EXPECT_EQ(vector.data(), data);
	EXPECT_EQ(vector.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(vector.begin(), vector.end()), (std::vector<int>{1, 2, 3}));
}

TEST(VectorTraits, NumericTypesAreSupported)
{
	static_assert(std::default_initializable<Vector<int>>);
	static_assert(std::default_initializable<Vector<float>>);
	static_assert(std::default_initializable<Vector<double>>);
	static_assert(std::is_copy_constructible_v<Vector<int>>);
	static_assert(std::is_move_constructible_v<Vector<int>>);
	static_assert(std::is_copy_assignable_v<Vector<int>>);
	static_assert(std::is_move_assignable_v<Vector<int>>);
	static_assert(std::is_nothrow_swappable_v<Vector<int>>);
}
