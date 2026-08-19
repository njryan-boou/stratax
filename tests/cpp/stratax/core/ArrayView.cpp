#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>

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
	EXPECT_EQ(view.size(), 6);
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
	EXPECT_EQ(view.size(), 0);
}

TEST(ArrayViewTypes, Aliases)
{
	using View = ArrayView<int>;

	static_assert(std::same_as<View::value_type, int>);
	static_assert(std::same_as<View::size_type, std::size_t>);
	static_assert(std::same_as<View::reference, int&>);
	static_assert(std::same_as<View::const_reference, const int&>);
	static_assert(std::same_as<View::pointer, int*>);
	static_assert(std::same_as<View::const_pointer, const int*>);
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
