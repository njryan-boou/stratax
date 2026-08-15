#include <gtest/gtest.h>

#include <stratax.h>

TEST(IndexingSlicing, vector_slice_keeps_step_order)
{
    const stratax::Vector<int> values{10, 20, 30, 40, 50};

    const auto stepped = stratax::indexing::slice(values, stratax::Slice{0, 5, 2});
    EXPECT_EQ(stepped.shape(), (stratax::Shape{3}));
    EXPECT_EQ(stepped[0], 10);
    EXPECT_EQ(stepped[1], 30);
    EXPECT_EQ(stepped[2], 50);

    const auto negative_bounds = stratax::indexing::slice(values, stratax::Slice{-3, -1});
    EXPECT_EQ(negative_bounds.shape(), (stratax::Shape{2}));
    EXPECT_EQ(negative_bounds[0], 30);
    EXPECT_EQ(negative_bounds[1], 40);
}

TEST(IndexingSlicing, matrix_slice_supports_row_and_column_steps)
{
    const stratax::Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    const auto sliced = stratax::indexing::slice(
        matrix,
        stratax::Slice{1, 3},
        stratax::Slice{0, 3, 2});

    EXPECT_EQ(sliced.shape(), (stratax::Shape{2, 2}));
    EXPECT_EQ(sliced(0, 0), 4);
    EXPECT_EQ(sliced(0, 1), 6);
    EXPECT_EQ(sliced(1, 0), 7);
    EXPECT_EQ(sliced(1, 1), 9);
}

TEST(IndexingSlicing, tensor_slice_supports_varargs_and_vector_ranges)
{
    stratax::Tensor<int> tensor(stratax::Shape{2, 2, 2});
    for (std::size_t i = 0; i < tensor.size(); ++i)
    {
        tensor[i] = static_cast<int>(i);
    }

    const auto from_varargs = stratax::indexing::slice(
        tensor,
        stratax::Slice{0, 2},
        stratax::Slice{0, 2},
        stratax::Slice{1, 2});

    EXPECT_EQ(from_varargs.shape(), (stratax::Shape{2, 2, 1}));
    EXPECT_EQ(from_varargs[0], 1);
    EXPECT_EQ(from_varargs[1], 3);
    EXPECT_EQ(from_varargs[2], 5);
    EXPECT_EQ(from_varargs[3], 7);

    const std::vector<stratax::Slice> ranges{
        stratax::Slice{0, 2},
        stratax::Slice{1, 2},
        stratax::Slice{0, 2, 2}};

    const auto from_vector = stratax::indexing::slice(tensor, ranges);
    EXPECT_EQ(from_vector.shape(), (stratax::Shape{2, 1, 1}));
    EXPECT_EQ(from_vector[0], 2);
    EXPECT_EQ(from_vector[1], 6);
}

TEST(IndexingSlicing, mismatched_rank_throws)
{
    const stratax::Tensor<int> tensor(stratax::Shape{2, 2, 2});
    const std::vector<stratax::Slice> wrong_rank{
        stratax::Slice{0, 2},
        stratax::Slice{0, 2}};

    EXPECT_THROW(([]() {
        const stratax::Tensor<int> tensor(stratax::Shape{2, 2, 2});
        const std::vector<stratax::Slice> wrong_rank{
            stratax::Slice{0, 2},
            stratax::Slice{0, 2}};
        stratax::indexing::slice(tensor, wrong_rank);
    })(), Exceptions::IndexError);

    EXPECT_THROW(([]() {
        const stratax::Vector<int> values{1, 2, 3};
        stratax::indexing::slice(values, stratax::Slice{0, 3, 2});
    })(), Exceptions::IndexError);
}

TEST(IndexingSlicing, empty_slice_returns_empty_container)
{
    const stratax::Vector<int> values{1, 2, 3};

    const auto empty = stratax::indexing::slice(values, stratax::Slice{3, 1});
    EXPECT_EQ(empty.shape(), (stratax::Shape{0}));
    EXPECT_TRUE(empty.empty());

    const auto empty_matrix = stratax::indexing::slice(
        stratax::Matrix<int>{{1, 2}, {3, 4}},
        stratax::Slice{2, 1},
        stratax::Slice{0, 2});
    EXPECT_EQ(empty_matrix.shape(), (stratax::Shape{0, 2}));
    EXPECT_TRUE(empty_matrix.empty());
}
