#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <stratax.h>

using namespace stratax::core;
using namespace stratax::container;

TEST(ShapeInvariant, ZeroExtentWinsOverOverflowInEveryPosition)
{
    const auto maximum = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> dims{0, 2, maximum};
    do {
        EXPECT_EQ(Shape(dims).elements(), 0);
    } while (std::next_permutation(dims.begin(), dims.end()));
}

TEST(SliceInvariant, NormalizationHandlesExtremeSteps)
{
    using namespace stratax::indexing::detail;
    const auto maximum = std::numeric_limits<std::ptrdiff_t>::max();
    const auto minimum = std::numeric_limits<std::ptrdiff_t>::min();
    EXPECT_EQ(normalize_slice(Slice{0, 5, maximum}, 5).size, 1);
    EXPECT_EQ(normalize_slice(Slice{4, -1, -maximum}, 5).size, 1);
    EXPECT_EQ(normalize_slice(Slice{4, -1, minimum}, 5).size, 1);
}

TEST(SliceInvariant, RawRangesHandleFullSignedDomain)
{
    const auto maximum = std::numeric_limits<std::ptrdiff_t>::max();
    const auto minimum = std::numeric_limits<std::ptrdiff_t>::min();
    const auto full_distance = std::numeric_limits<std::size_t>::max();
    EXPECT_EQ((Slice{minimum, maximum, 1}.size()), full_distance);
    EXPECT_EQ((Slice{maximum, minimum, -1}.size()), full_distance);
    EXPECT_EQ((Slice{maximum, minimum, minimum}.size()), 2);
    EXPECT_EQ((Slice{0, 5, maximum}.size()), 1);
}

TEST(BroadcastInvariant, EmptyRankZeroOperandCannotSupplyValues)
{
    const Tensor<int> empty(Shape{});
    const Tensor<int> values(Shape{2, 3}, 7);
    EXPECT_THROW(static_cast<void>(empty + values), Exceptions::BroadcastError);
    EXPECT_THROW(static_cast<void>(values + empty), Exceptions::BroadcastError);
    EXPECT_THROW(static_cast<void>(empty == values), Exceptions::BroadcastError);
    EXPECT_THROW(static_cast<void>(empty & values), Exceptions::BroadcastError);
    EXPECT_THROW(static_cast<void>(values << empty), Exceptions::BroadcastError);
    auto mutable_values = values;
    EXPECT_THROW(mutable_values += empty, Exceptions::BroadcastError);
    EXPECT_THROW(mutable_values &= empty, Exceptions::BroadcastError);
    EXPECT_THROW(mutable_values <<= empty, Exceptions::BroadcastError);
    EXPECT_EQ(mutable_values[0], 7);
    EXPECT_TRUE((empty + empty).empty());
    EXPECT_TRUE((empty + Tensor<int>(Shape{0, 3})).empty());
}

TEST(ArrayViewInvariant, CheckedAccessAndIteratorArithmeticUseLogicalOrder)
{
    int data[]{0, 1, 2, 3, 4, 5};
    ArrayView<int> view(data, Shape{3, 2}, Shape{1, 3});
    EXPECT_EQ(view.at(-1), 5);
    EXPECT_EQ(view.at(std::vector<std::ptrdiff_t>{-1, -2}), 2);
    EXPECT_THROW(view.at(6), Exceptions::IndexError);
    EXPECT_THROW(view.at(-7), Exceptions::IndexError);
    EXPECT_THROW(view.at(std::vector<std::ptrdiff_t>{3, 0}), Exceptions::IndexError);
    EXPECT_THROW(view.at(std::vector<std::ptrdiff_t>{0}), Exceptions::RankError);
    EXPECT_EQ(view.end() - view.begin(), 6);
    EXPECT_EQ(*(2 + view.begin()), 1);
    EXPECT_EQ(view.end()[-2], 2);
    ArrayView<int>::const_iterator read_only = view.begin();
    EXPECT_EQ(read_only, view.cbegin());
    std::sort(view.begin(), view.end());
    EXPECT_EQ(std::vector<int>(view.begin(), view.end()), (std::vector<int>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(data[1], 2);
}

template<typename A>
void expect_layout(const A& array)
{
    EXPECT_EQ(array.size(), array.shape().elements());
    EXPECT_EQ(array.rank(), array.strides().rank());
    EXPECT_EQ(array.strides(), array.shape().strides());
    EXPECT_EQ(array.data() == nullptr, array.empty());
}

template<typename A>
void check_move_invariants(A source, const Shape& empty_shape)
{
    const auto original_shape = source.shape();
    auto* storage = source.data();
    A destination(std::move(source));
    expect_layout(destination);
    expect_layout(source);
    EXPECT_EQ(destination.data(), storage);
    EXPECT_EQ(destination.shape(), original_shape);
    EXPECT_EQ(source.shape(), empty_shape);

    source = std::move(destination);
    expect_layout(source);
    expect_layout(destination);
    EXPECT_EQ(source.data(), storage);
    EXPECT_EQ(destination.shape(), empty_shape);
}

TEST(ContainerInvariant, VectorMovesPreserveRank)
{
    check_move_invariants(Vector<int>{1, 2, 3}, Shape{0});
}

TEST(ContainerInvariant, MatrixMovesPreserveRank)
{
    check_move_invariants(Matrix<int>{{1, 2}, {3, 4}}, Shape{0, 0});
}

TEST(SliceInvariant, EmptyViewsKeepAValidBasePointer)
{
    Matrix<int> matrix{{1, 2}, {3, 4}};
    auto matrix_view = stratax::indexing::slice(matrix, Slice{2, 2}, Slice{2, 2});
    EXPECT_TRUE(matrix_view.empty());
    EXPECT_EQ(matrix_view.data(), matrix.data());

    Tensor<int> tensor(Shape{2, 2}, 1);
    auto tensor_view = stratax::indexing::slice(tensor, Slice{2, 2}, Slice{2, 2});
    auto dynamic_view = stratax::indexing::slice(tensor, std::vector<Slice>{{2, 2}, {2, 2}});
    EXPECT_TRUE(tensor_view.empty());
    EXPECT_TRUE(dynamic_view.empty());
    EXPECT_EQ(tensor_view.data(), tensor.data());
    EXPECT_EQ(dynamic_view.data(), tensor.data());

    Vector<int> empty;
    EXPECT_EQ(stratax::indexing::slice(empty, Slice{0, 0}).data(), nullptr);
}

TEST(ArrayViewInvariant, RejectsInvalidMetadataAtConstruction)
{
    int value = 1;
    using stratax::core::ArrayView;
    EXPECT_THROW((ArrayView<int>(&value, Shape{1, 1}, Shape{1})), Exceptions::RankError);
    EXPECT_THROW((ArrayView<int>(nullptr, Shape{1}, Shape{1})), Exceptions::ValueError);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW((ArrayView<int>(&value, Shape{maximum, 2}, Shape{2, 1})), Exceptions::DimensionError);
    EXPECT_THROW((ArrayView<int>(&value, Shape{3}, Shape{maximum})), Exceptions::DimensionError);
    EXPECT_THROW((ArrayView<int>(&value, Shape{2, 2}, Shape{maximum, 1})), Exceptions::DimensionError);
}
