#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include <stratax.h>

using stratax::container::Tensor;
using stratax::core::ArrayView;
using stratax::core::Shape;

namespace {

template<typename Array>
void expect_empty_checked_access(Array& array,
                                 const std::vector<std::ptrdiff_t>& coordinates)
{
    const Array& read_only = array;
    EXPECT_THROW(array.at(coordinates), Exceptions::IndexError);
    EXPECT_THROW(read_only.at(coordinates), Exceptions::IndexError);
    EXPECT_THROW(array.at(coordinates) = 7, Exceptions::IndexError);
    EXPECT_THROW(array.at(0), Exceptions::IndexError);
    EXPECT_THROW(read_only.at(-1), Exceptions::IndexError);
}

} // namespace

TEST(CheckedEmptyIndexing, RankZeroTensorHasNoScalarElement)
{
    Tensor<int> tensor(Shape{});
    expect_empty_checked_access(tensor, {});
    EXPECT_THROW(tensor.at(std::vector<std::ptrdiff_t>{0}), Exceptions::RankError);
    EXPECT_THROW(std::as_const(tensor).at(std::vector<std::ptrdiff_t>{0}),
                 Exceptions::RankError);
}

TEST(CheckedEmptyIndexing, MovedFromTensorHasNoScalarElement)
{
    Tensor<int> source(Shape{2}, 9);
    Tensor<int> destination(std::move(source));
    expect_empty_checked_access(source, {});
    EXPECT_EQ(destination.at(std::vector<std::ptrdiff_t>{-1}), 9);
}

TEST(CheckedEmptyIndexing, RankZeroNullViewHasNoScalarElement)
{
    ArrayView<int> view(nullptr, Shape{}, Shape{});
    expect_empty_checked_access(view, {});
    EXPECT_THROW(view.at(std::vector<std::ptrdiff_t>{0}), Exceptions::RankError);
    EXPECT_THROW(std::as_const(view).at(std::vector<std::ptrdiff_t>{0}),
                 Exceptions::RankError);
}

TEST(CheckedEmptyIndexing, RankZeroViewCannotExposeNonNullBacking)
{
    int backing = 42;
    ArrayView<int> view(&backing, Shape{}, Shape{});
    expect_empty_checked_access(view, {});
    ArrayView<const int> immutable(&backing, Shape{}, Shape{});
    EXPECT_THROW(immutable.at(std::vector<std::ptrdiff_t>{}), Exceptions::IndexError);
    EXPECT_THROW(std::as_const(immutable).at(std::vector<std::ptrdiff_t>{}),
                 Exceptions::IndexError);
    EXPECT_EQ(backing, 42);
}

TEST(CheckedEmptyIndexing, ZeroExtentRejectsMatchingRankButPreservesRankErrors)
{
    Tensor<int> tensor(Shape{2, 0, 3});
    expect_empty_checked_access(tensor, {0, 0, 0});
    EXPECT_THROW(tensor.at(std::vector<std::ptrdiff_t>{}), Exceptions::RankError);
    EXPECT_THROW(std::as_const(tensor).at(std::vector<std::ptrdiff_t>{}),
                 Exceptions::RankError);

    int backing = 42;
    ArrayView<int> view(&backing, Shape{2, 0, 3}, Shape{0, 3, 1});
    expect_empty_checked_access(view, {0, 0, 0});
    EXPECT_THROW(view.at(std::vector<std::ptrdiff_t>{}), Exceptions::RankError);
    EXPECT_THROW(std::as_const(view).at(std::vector<std::ptrdiff_t>{}),
                 Exceptions::RankError);
    EXPECT_EQ(backing, 42);
}

TEST(CheckedEmptyIndexing, NonemptyCheckedAccessRetainsNegativeIndices)
{
    Tensor<int> tensor(Shape{2, 3}, 0);
    tensor.at(std::vector<std::ptrdiff_t>{-1, -1}) = 17;
    EXPECT_EQ(std::as_const(tensor).at(std::vector<std::ptrdiff_t>{1, 2}), 17);
    ArrayView<int> view(tensor.data(), tensor.shape(), tensor.strides());
    view.at(std::vector<std::ptrdiff_t>{-2, -3}) = 23;
    EXPECT_EQ(std::as_const(view).at(std::vector<std::ptrdiff_t>{0, 0}), 23);
}
