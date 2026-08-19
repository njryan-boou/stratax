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

TEST(TensorConstructor, DefaultConstructor)
{
	const Tensor<int> tensor;

	EXPECT_TRUE(tensor.empty());
	EXPECT_EQ(tensor.size(), 0);
	EXPECT_EQ(tensor.rank(), 1);
	EXPECT_EQ(tensor.shape(), Shape({0}));
	EXPECT_EQ(tensor.strides(), Shape({1}));
}

TEST(TensorConstructor, RankZeroShape)
{
	const Tensor<int> tensor(Shape{});

	EXPECT_TRUE(tensor.empty());
	EXPECT_EQ(tensor.rank(), 0);
	EXPECT_EQ(tensor.shape(), Shape{});
	EXPECT_TRUE(tensor.strides().empty());
}

TEST(TensorConstructor, RankOneShape)
{
	const Tensor<int> tensor(Shape{4});

	EXPECT_EQ(tensor.size(), 4);
	EXPECT_EQ(tensor.rank(), 1);
	EXPECT_EQ(tensor.shape(), Shape({4}));
	EXPECT_EQ(tensor.strides(), Shape({1}));
}

TEST(TensorConstructor, MultidimensionalShape)
{
	const Tensor<int> tensor(Shape{2, 3, 4});

	EXPECT_EQ(tensor.size(), 24);
	EXPECT_EQ(tensor.rank(), 3);
	EXPECT_EQ(tensor.shape(), Shape({2, 3, 4}));
	EXPECT_EQ(tensor.strides(), Shape({12, 4, 1}));

	for (const int value : tensor) {
		EXPECT_EQ(value, 0);
	}
}

TEST(TensorConstructor, ZeroDimension)
{
	const Tensor<int> tensor(Shape{2, 0, 4});

	EXPECT_TRUE(tensor.empty());
	EXPECT_EQ(tensor.rank(), 3);
	EXPECT_EQ(tensor.shape(), Shape({2, 0, 4}));
	EXPECT_EQ(tensor.strides(), Shape({0, 4, 1}));
}

TEST(TensorConstructor, FillValue)
{
	const Tensor<int> tensor(Shape{2, 3, 4}, 7);

	EXPECT_EQ(tensor.size(), 24);
	for (const int value : tensor) {
		EXPECT_EQ(value, 7);
	}
}

TEST(TensorConstructor, RejectsElementCountOverflow)
{
	EXPECT_THROW(
		static_cast<void>(Tensor<int>(
			Shape{std::numeric_limits<std::size_t>::max(), 2})),
		Exceptions::DimensionError
	);
}

TEST(TensorConstructor, Copy)
{
	Tensor<int> original(Shape{2, 2});
	for (std::size_t i = 0; i < original.size(); ++i) {
		original[i] = static_cast<int>(i + 1);
	}

	Tensor<int> copy(original);

	EXPECT_EQ(copy.shape(), original.shape());
	EXPECT_EQ(std::vector<int>(copy.begin(), copy.end()),
		(std::vector<int>{1, 2, 3, 4}));
	EXPECT_NE(copy.data(), original.data());

	copy(0, 0) = 10;
	EXPECT_EQ(original(0, 0), 1);
}

TEST(TensorConstructor, Move)
{
	Tensor<int> source(Shape{2, 2}, 5);
	int* source_data = source.data();

	Tensor<int> destination(std::move(source));

	EXPECT_EQ(destination.data(), source_data);
	EXPECT_EQ(destination.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(destination.begin(), destination.end()),
		(std::vector<int>{5, 5, 5, 5}));
	EXPECT_TRUE(source.empty());
}

TEST(TensorAssignment, CopyAssignment)
{
	const Tensor<int> source(Shape{2, 2}, 5);
	Tensor<int> destination(Shape{3}, 9);

	destination = source;

	EXPECT_EQ(destination.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(destination.begin(), destination.end()),
		(std::vector<int>{5, 5, 5, 5}));
	EXPECT_NE(destination.data(), source.data());
}

TEST(TensorAssignment, SelfCopyAssignment)
{
	Tensor<int> tensor(Shape{2, 2}, 5);
	int* data = tensor.data();

	tensor = tensor;

	EXPECT_EQ(tensor.data(), data);
	EXPECT_EQ(tensor.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(tensor.begin(), tensor.end()),
		(std::vector<int>{5, 5, 5, 5}));
}

TEST(TensorAssignment, MoveAssignment)
{
	Tensor<int> source(Shape{2, 2}, 5);
	Tensor<int> destination(Shape{3}, 9);
	int* source_data = source.data();

	destination = std::move(source);

	EXPECT_EQ(destination.data(), source_data);
	EXPECT_EQ(destination.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(destination.begin(), destination.end()),
		(std::vector<int>{5, 5, 5, 5}));
	EXPECT_TRUE(source.empty());
}

TEST(TensorAssignment, SelfMoveAssignment)
{
	Tensor<int> tensor(Shape{2, 2}, 5);
	int* data = tensor.data();

	tensor = std::move(tensor);

	EXPECT_EQ(tensor.data(), data);
	EXPECT_EQ(tensor.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(tensor.begin(), tensor.end()),
		(std::vector<int>{5, 5, 5, 5}));
}

TEST(TensorTypes, Aliases)
{
	using TestTensor = Tensor<int>;

	static_assert(std::same_as<TestTensor::value_type, int>);
	static_assert(std::same_as<TestTensor::size_type, std::size_t>);
	static_assert(std::same_as<TestTensor::difference_type, std::ptrdiff_t>);
	static_assert(std::same_as<TestTensor::reference, int&>);
	static_assert(std::same_as<TestTensor::const_reference, const int&>);
	static_assert(std::same_as<TestTensor::pointer, int*>);
	static_assert(std::same_as<TestTensor::const_pointer, const int*>);
	static_assert(std::contiguous_iterator<TestTensor::iterator>);
	static_assert(std::contiguous_iterator<TestTensor::const_iterator>);
	static_assert(std::same_as<
		TestTensor::reverse_iterator,
		std::reverse_iterator<TestTensor::iterator>
	>);
	static_assert(std::same_as<
		TestTensor::const_reverse_iterator,
		std::reverse_iterator<TestTensor::const_iterator>
	>);
}

TEST(TensorAccess, UncheckedVariadicRowMajorLayout)
{
	Tensor<int> tensor(Shape{2, 3, 4});
	for (std::size_t i = 0; i < tensor.size(); ++i) {
		tensor[i] = static_cast<int>(i);
	}
	const Tensor<int>& const_tensor = tensor;

	EXPECT_EQ(tensor(0, 0, 0), 0);
	EXPECT_EQ(tensor(0, 2, 3), 11);
	EXPECT_EQ(tensor(1, 0, 0), 12);
	EXPECT_EQ(tensor(1, 2, 3), 23);

	tensor(1, 1, 2) = 99;
	EXPECT_EQ(const_tensor(1, 1, 2), 99);
	EXPECT_EQ(tensor[18], 99);
	static_assert(std::same_as<decltype(tensor(0, 0, 0)), int&>);
	static_assert(std::same_as<decltype(const_tensor(0, 0, 0)), const int&>);
}

TEST(TensorAccess, UncheckedRankOneVariadicIndex)
{
	Tensor<int> tensor(Shape{3}, 0);

	tensor(1) = 8;

	EXPECT_EQ(tensor[1], 8);
}

TEST(TensorAccess, UncheckedVectorIndices)
{
	Tensor<int> tensor(Shape{2, 3, 4});
	const std::vector<std::size_t> indices{1, 2, 3};
	const Tensor<int>& const_tensor = tensor;

	tensor(indices) = 8;

	EXPECT_EQ(tensor[23], 8);
	EXPECT_EQ(const_tensor(indices), 8);
}

TEST(TensorAccess, CheckedVariadicPositiveIndices)
{
	Tensor<int> tensor(Shape{2, 3, 4}, 0);
	const Tensor<int>& const_tensor = tensor;

	tensor.at(1, 2, 3) = 8;

	EXPECT_EQ(tensor[23], 8);
	EXPECT_EQ(tensor.at(1, 2, 3), 8);
	EXPECT_EQ(const_tensor.at(1, 2, 3), 8);
}

TEST(TensorAccess, CheckedVariadicNegativeIndices)
{
	Tensor<int> tensor(Shape{2, 3, 4});
	for (std::size_t i = 0; i < tensor.size(); ++i) {
		tensor[i] = static_cast<int>(i);
	}

	EXPECT_EQ(tensor.at(-1, -1, -1), 23);
	EXPECT_EQ(tensor.at(-2, -3, -4), 0);
	EXPECT_EQ(tensor.at(-1, 0, -2), 14);
}

TEST(TensorAccess, CheckedVectorIndices)
{
	Tensor<int> tensor(Shape{2, 3, 4});
	for (std::size_t i = 0; i < tensor.size(); ++i) {
		tensor[i] = static_cast<int>(i);
	}
	const Tensor<int>& const_tensor = tensor;
	const std::vector<std::ptrdiff_t> positive{1, 2, 3};
	const std::vector<std::ptrdiff_t> negative{-1, -1, -1};

	EXPECT_EQ(tensor.at(positive), 23);
	EXPECT_EQ(const_tensor.at(negative), 23);
	tensor.at(negative) = 99;
	EXPECT_EQ(tensor[23], 99);
}

TEST(TensorAccess, InheritedFlatAt)
{
	Tensor<int> tensor(Shape{2, 3}, 0);
	const Tensor<int>& const_tensor = tensor;

	tensor.at(4) = 8;

	EXPECT_EQ(tensor.at(4), 8);
	EXPECT_EQ(const_tensor.at(-2), 8);
	EXPECT_THROW(static_cast<void>(tensor.at(6)), Exceptions::IndexError);
}

TEST(TensorAccess, VariadicRankMismatch)
{
	const Tensor<int> tensor(Shape{2, 3, 4});

	EXPECT_THROW(static_cast<void>(tensor.at(1, 2)), Exceptions::IndexError);
	EXPECT_THROW(static_cast<void>(tensor.at(1, 2, 3, 0)), Exceptions::IndexError);
}

TEST(TensorAccess, VectorRankMismatch)
{
	const Tensor<int> tensor(Shape{2, 3, 4});

	EXPECT_THROW(
		static_cast<void>(tensor.at(std::vector<std::ptrdiff_t>{1, 2})),
		Exceptions::IndexError
	);
}

TEST(TensorAccess, ComponentOutOfRange)
{
	const Tensor<int> tensor(Shape{2, 3, 4});

	EXPECT_THROW(static_cast<void>(tensor.at(2, 0, 0)), Exceptions::IndexError);
	EXPECT_THROW(static_cast<void>(tensor.at(-3, 0, 0)), Exceptions::IndexError);
	EXPECT_THROW(static_cast<void>(tensor.at(0, 3, 0)), Exceptions::IndexError);
	EXPECT_THROW(static_cast<void>(tensor.at(0, 0, -5)), Exceptions::IndexError);
}

TEST(TensorAccess, RankMismatchErrorMessage)
{
	const Tensor<int> tensor(Shape{2, 3, 4});

	try {
		static_cast<void>(tensor.at(1, 2));
		FAIL() << "Expected Exceptions::IndexError";
	} catch (const Exceptions::IndexError& error) {
		EXPECT_STREQ(
			error.what(),
			"Tensor multi-index rank must match tensor rank."
		);
	}
}

TEST(TensorAccess, ComponentOutOfRangeErrorMessage)
{
	const Tensor<int> tensor(Shape{2, 3, 4});

	try {
		static_cast<void>(tensor.at(0, 3, 0));
		FAIL() << "Expected Exceptions::IndexError";
	} catch (const Exceptions::IndexError& error) {
		EXPECT_STREQ(
			error.what(),
			"Tensor multi-index component is out of bounds."
		);
	}
}

TEST(TensorIterator, ForwardAndReverseTraversal)
{
	Tensor<int> tensor(Shape{2, 2});
	for (std::size_t i = 0; i < tensor.size(); ++i) {
		tensor[i] = static_cast<int>(i + 1);
	}

	EXPECT_EQ(std::vector<int>(tensor.begin(), tensor.end()),
		(std::vector<int>{1, 2, 3, 4}));
	EXPECT_EQ(std::vector<int>(tensor.crbegin(), tensor.crend()),
		(std::vector<int>{4, 3, 2, 1}));

	for (int& value : tensor) {
		value *= 2;
	}
	EXPECT_EQ(std::vector<int>(tensor.cbegin(), tensor.cend()),
		(std::vector<int>{2, 4, 6, 8}));
}

TEST(TensorModifier, Fill)
{
	Tensor<int> tensor(Shape{2, 2}, 1);

	tensor.fill(9);

	EXPECT_EQ(std::vector<int>(tensor.begin(), tensor.end()),
		(std::vector<int>{9, 9, 9, 9}));
}

TEST(TensorModifier, MemberSwap)
{
	Tensor<int> lhs(Shape{2, 2}, 1);
	Tensor<int> rhs(Shape{2, 1, 2}, 7);
	int* lhs_data = lhs.data();
	int* rhs_data = rhs.data();

	lhs.swap(rhs);

	EXPECT_EQ(lhs.data(), rhs_data);
	EXPECT_EQ(rhs.data(), lhs_data);
	EXPECT_EQ(lhs.shape(), Shape({2, 1, 2}));
	EXPECT_EQ(rhs.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()),
		(std::vector<int>{7, 7, 7, 7}));
	EXPECT_EQ(std::vector<int>(rhs.begin(), rhs.end()),
		(std::vector<int>{1, 1, 1, 1}));
}

TEST(TensorModifier, NonMemberSwap)
{
	Tensor<int> lhs(Shape{2, 2}, 1);
	Tensor<int> rhs(Shape{3}, 7);

	using std::swap;
	swap(lhs, rhs);

	EXPECT_EQ(lhs.shape(), Shape({3}));
	EXPECT_EQ(rhs.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()),
		(std::vector<int>{7, 7, 7}));
	EXPECT_EQ(std::vector<int>(rhs.begin(), rhs.end()),
		(std::vector<int>{1, 1, 1, 1}));
}

TEST(TensorModifier, SelfSwap)
{
	Tensor<int> tensor(Shape{2, 2}, 5);
	int* data = tensor.data();

	tensor.swap(tensor);

	EXPECT_EQ(tensor.data(), data);
	EXPECT_EQ(tensor.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(tensor.begin(), tensor.end()),
		(std::vector<int>{5, 5, 5, 5}));
}

TEST(TensorTraits, NumericTypesAreSupported)
{
	static_assert(std::default_initializable<Tensor<int>>);
	static_assert(std::default_initializable<Tensor<float>>);
	static_assert(std::default_initializable<Tensor<double>>);
	static_assert(std::is_copy_constructible_v<Tensor<int>>);
	static_assert(std::is_move_constructible_v<Tensor<int>>);
	static_assert(std::is_copy_assignable_v<Tensor<int>>);
	static_assert(std::is_move_assignable_v<Tensor<int>>);
	static_assert(std::is_nothrow_swappable_v<Tensor<int>>);
}
