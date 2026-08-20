#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <vector>

#include <stratax.h>

using namespace stratax::core;
using namespace stratax::container;

TEST(ArrayViewConstructor, PreservesMetadata)
{
	int values[]{1, 2, 3, 4, 5, 6};
	const Shape shape{2, 3};
	const Shape strides{3, 1};
	const ArrayView<int> view(values, shape, strides);

	EXPECT_EQ(view.data(), values);
	EXPECT_EQ(view.shape(), shape);
	EXPECT_EQ(view.strides(), strides);
	EXPECT_EQ(view.ndim(), 2);
	EXPECT_EQ(view.rank(), 2);
	EXPECT_EQ(view.size(), 6);
	EXPECT_FALSE(view.empty());
}

TEST(ArrayViewConstructor, PreservesCustomStrides)
{
	int values[]{1, 2, 3, 4, 5, 6};
	const ArrayView<int> view(values, Shape{3, 2}, Shape{1, 3});

	EXPECT_EQ(view.shape(), (Shape{3, 2}));
	EXPECT_EQ(view.strides(), (Shape{1, 3}));
}

TEST(ArrayViewConstructor, EmptyView)
{
	const ArrayView<int> view(nullptr, Shape{}, Shape{});

	EXPECT_EQ(view.data(), nullptr);
	EXPECT_TRUE(view.shape().empty());
	EXPECT_TRUE(view.strides().empty());
	EXPECT_EQ(view.ndim(), 0);
	EXPECT_EQ(view.rank(), 0);
	EXPECT_EQ(view.size(), 0);
	EXPECT_TRUE(view.empty());
}

TEST(ArrayViewTypes, Aliases)
{
	using View = ArrayView<int>;

	static_assert(std::same_as<View::value_type, int>);
	static_assert(std::same_as<View::element_type, int>);
	static_assert(std::same_as<View::size_type, std::size_t>);
	static_assert(std::same_as<View::reference, int&>);
	static_assert(std::same_as<View::const_reference, const int&>);
	static_assert(std::same_as<View::pointer, int*>);
	static_assert(std::same_as<View::const_pointer, const int*>);
	static_assert(std::same_as<View::const_view, ArrayView<const int>>);
}

TEST(ArrayViewTypes, ConstViewAliases)
{
	using View = ArrayView<const int>;

	static_assert(std::same_as<View::value_type, int>);
	static_assert(std::same_as<View::element_type, const int>);
	static_assert(std::same_as<View::reference, const int&>);
	static_assert(std::same_as<View::const_reference, const int&>);
	static_assert(std::same_as<View::pointer, const int*>);
	static_assert(std::same_as<View::const_pointer, const int*>);
	static_assert(std::constructible_from<ArrayView<const int>, ArrayView<int>>);
	static_assert(!std::constructible_from<ArrayView<int>, ArrayView<const int>>);
}

TEST(ArrayViewData, ProvidesMutableAccess)
{
	int values[]{1, 2, 3};
	ArrayView<int> view(values, Shape{3}, Shape{1});

	view.data()[1] = 20;

	EXPECT_EQ(values[1], 20);
	EXPECT_EQ(view.data()[1], 20);
}

TEST(ArrayViewData, ProvidesConstAccess)
{
	int values[]{1, 2, 3};
	const ArrayView<int> view(values, Shape{3}, Shape{1});

	static_assert(
		std::same_as<decltype(view.data()), ArrayView<int>::const_pointer>);
	EXPECT_EQ(view.data()[2], 3);
}

TEST(ArrayViewData, ConvertsToConstView)
{
	int values[]{1, 2, 3, 4};
	const ArrayView<int> mutable_view(values, Shape{2, 2}, Shape{2, 1});
	const ArrayView<const int> converted = mutable_view;
	const auto explicit_const = mutable_view.as_const();

	static_assert(std::same_as<
		decltype(explicit_const),
		const ArrayView<const int>>);
	static_assert(std::same_as<decltype(converted(0, 0)), const int&>);
	EXPECT_EQ(converted.data(), values);
	EXPECT_EQ(converted.shape(), mutable_view.shape());
	EXPECT_EQ(converted.strides(), mutable_view.strides());
	EXPECT_EQ(converted(1, 1), 4);
	EXPECT_EQ(explicit_const(0, 1), 2);
}

TEST(ArrayViewOwnership, ReferencesExternalStorage)
{
	int values[]{1, 2, 3};
	const ArrayView<int> view(values, Shape{3}, Shape{1});

	values[0] = 10;

	EXPECT_EQ(view.data()[0], 10);
}

TEST(ArrayViewIndexing, UsesStoredStrides)
{
	int values[]{1, 2, 3, 4, 5, 6};
	const ArrayView<int> view(values, Shape{3, 2}, Shape{1, 3});

	EXPECT_EQ(view(0, 0), 1);
	EXPECT_EQ(view(0, 1), 4);
	EXPECT_EQ(view(1, 0), 2);
	EXPECT_EQ(view(2, 1), 6);
}

TEST(ArrayViewIndexing, ProvidesMutableAccess)
{
	int values[]{1, 2, 3, 4, 5, 6};
	ArrayView<int> view(values, Shape{3, 2}, Shape{1, 3});

	view(1, 1) = 40;

	EXPECT_EQ(values[4], 40);
}

TEST(ArrayViewIndexing, ProvidesConstAccess)
{
	int values[]{1, 2, 3, 4};
	const ArrayView<int> view(values, Shape{2, 2}, Shape{2, 1});

	static_assert(std::same_as<decltype(view(0, 0)), const int&>);
	EXPECT_EQ(view(1, 0), 3);
}

TEST(ArrayViewFlatIndexing, UsesLogicalRowMajorOrder)
{
	int values[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
	ArrayView<int> view(values + 1, Shape{2, 2}, Shape{6, 2});

	EXPECT_EQ(view[0], 1);
	EXPECT_EQ(view[1], 3);
	EXPECT_EQ(view[2], 7);
	EXPECT_EQ(view[3], 9);

	view[2] = 70;
	EXPECT_EQ(values[7], 70);
}

TEST(ArrayViewIteration, TraversesLogicalRowMajorOrder)
{
	int values[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
	ArrayView<int> view(values + 1, Shape{2, 2}, Shape{6, 2});
	const std::vector<int> expected{1, 3, 7, 9};

	static_assert(std::random_access_iterator<ArrayView<int>::iterator>);
	static_assert(std::random_access_iterator<ArrayView<int>::const_iterator>);
	EXPECT_EQ(std::vector<int>(view.begin(), view.end()), expected);
	EXPECT_EQ(std::vector<int>(view.cbegin(), view.cend()), expected);
}

TEST(ArrayView, ShiftedView)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    ArrayView<int> view{
        matrix.data() + 1,
        Shape{2, 2},
        Shape{3, 1}
    };

    EXPECT_EQ(view(0, 0), 2);
    EXPECT_EQ(view(0, 1), 3);
    EXPECT_EQ(view(1, 0), 5);
    EXPECT_EQ(view(1, 1), 6);

    view(1, 0) = 50;

    EXPECT_EQ(matrix(1, 1), 50);
}
