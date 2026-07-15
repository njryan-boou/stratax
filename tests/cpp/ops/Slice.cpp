#include <gtest/gtest.h>
#include <complex>

#include <stratax.h>

using namespace stratax::container;
using stratax::core::Shape;
using stratax::core::Slice;

TEST(OpsSlice, vector_slice)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    auto result = slice(vector, Slice{1, 4});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result.size() == 3);
    EXPECT_TRUE(result[0] == 2);
    EXPECT_TRUE(result[1] == 3);
    EXPECT_TRUE(result[2] == 4);
}

TEST(OpsSlice, vector_slice_with_step)
{
    Vector<int> vector{1, 2, 3, 4, 5, 6};

    auto result = slice(vector, Slice{0, 6, 2});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result.size() == 3);
    EXPECT_TRUE(result[0] == 1);
    EXPECT_TRUE(result[1] == 3);
    EXPECT_TRUE(result[2] == 5);
}

TEST(OpsSlice, vector_empty_slice)
{
    Vector<int> vector{1, 2, 3};

    auto result = slice(vector, Slice{2, 2});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 0);
    EXPECT_TRUE(result.size() == 0);
    EXPECT_TRUE(result.empty());
}

TEST(OpsSlice, vector_slice_clamps_out_of_bounds)
{
    Vector<int> vector{1, 2, 3};

    auto result = slice(vector, Slice{0, 4});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result.size() == 3);
    EXPECT_TRUE(result[0] == 1);
    EXPECT_TRUE(result[1] == 2);
    EXPECT_TRUE(result[2] == 3);
}

TEST(OpsSlice, vector_slice_supports_negative_step)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    auto result = slice(vector, Slice{-1, -6, -2});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result.size() == 3);
    EXPECT_TRUE(result[0] == 5);
    EXPECT_TRUE(result[1] == 3);
    EXPECT_TRUE(result[2] == 1);
}

TEST(OpsSlice, matrix_slice)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    auto result = slice(matrix, Slice{1, 3}, Slice{0, 2});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 2);
    EXPECT_TRUE(result.cols() == 2);
    EXPECT_TRUE(result(0, 0) == 4);
    EXPECT_TRUE(result(0, 1) == 5);
    EXPECT_TRUE(result(1, 0) == 7);
    EXPECT_TRUE(result(1, 1) == 8);
}

TEST(OpsSlice, matrix_slice_with_step)
{
    Matrix<int> matrix{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    auto result = slice(matrix, Slice{0, 3, 2}, Slice{0, 4, 2});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 2);
    EXPECT_TRUE(result.cols() == 2);
    EXPECT_TRUE(result(0, 0) == 1);
    EXPECT_TRUE(result(0, 1) == 3);
    EXPECT_TRUE(result(1, 0) == 9);
    EXPECT_TRUE(result(1, 1) == 11);
}

TEST(OpsSlice, matrix_empty_slice)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    auto result = slice(matrix, Slice{0, 2}, Slice{1, 1});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 2);
    EXPECT_TRUE(result.cols() == 0);
    EXPECT_TRUE(result.size() == 0);
    EXPECT_TRUE(result.empty());
}

TEST(OpsSlice, matrix_slice_clamps_out_of_bounds)
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    auto row_clamped = slice(matrix, Slice{0, 3}, Slice{0, 2});
    auto col_clamped = slice(matrix, Slice{0, 2}, Slice{0, 3});

    EXPECT_TRUE(row_clamped.rank() == 2);
    EXPECT_TRUE(row_clamped.rows() == 2);
    EXPECT_TRUE(row_clamped.cols() == 2);
    EXPECT_TRUE(row_clamped(0, 0) == 1);
    EXPECT_TRUE(row_clamped(0, 1) == 2);
    EXPECT_TRUE(row_clamped(1, 0) == 3);
    EXPECT_TRUE(row_clamped(1, 1) == 4);

    EXPECT_TRUE(col_clamped.rank() == 2);
    EXPECT_TRUE(col_clamped.rows() == 2);
    EXPECT_TRUE(col_clamped.cols() == 2);
    EXPECT_TRUE(col_clamped(0, 0) == 1);
    EXPECT_TRUE(col_clamped(0, 1) == 2);
    EXPECT_TRUE(col_clamped(1, 0) == 3);
    EXPECT_TRUE(col_clamped(1, 1) == 4);
}

TEST(OpsSlice, matrix_slice_supports_negative_step)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    auto result = slice(matrix, Slice{-1, -3, -1}, Slice{-1, -4, -2});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 2);
    EXPECT_TRUE(result.cols() == 2);
    EXPECT_TRUE(result(0, 0) == 6);
    EXPECT_TRUE(result(0, 1) == 4);
    EXPECT_TRUE(result(1, 0) == 3);
    EXPECT_TRUE(result(1, 1) == 1);
}

TEST(OpsSlice, tensor_rank_one_slice)
{
    Tensor<int> tensor(Shape{5});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto result = slice(tensor, Slice{1, 4});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result.size() == 3);
    EXPECT_TRUE(result[0] == 2);
    EXPECT_TRUE(result[1] == 3);
    EXPECT_TRUE(result[2] == 4);
}

TEST(OpsSlice, tensor_complex_slice)
{
    Tensor<std::complex<double>> tensor(Shape{3});

    tensor[0] = {1.0, 2.0};
    tensor[1] = {3.0, 4.0};
    tensor[2] = {5.0, 6.0};

    auto result = slice(tensor, Slice{1, 3});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result[0] == std::complex<double>(3.0, 4.0));
    EXPECT_TRUE(result[1] == std::complex<double>(5.0, 6.0));
}

TEST(OpsSlice, tensor_rank_two_slice)
{
    Tensor<int> tensor(Shape{3, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto result = slice(tensor, Slice{1, 3}, Slice{1, 4});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 3);
    EXPECT_TRUE(result.size() == 6);

    EXPECT_TRUE(result(0, 0) == 6);
    EXPECT_TRUE(result(0, 1) == 7);
    EXPECT_TRUE(result(0, 2) == 8);
    EXPECT_TRUE(result(1, 0) == 10);
    EXPECT_TRUE(result(1, 1) == 11);
    EXPECT_TRUE(result(1, 2) == 12);
}

TEST(OpsSlice, tensor_rank_two_slice_with_step)
{
    Tensor<int> tensor(Shape{3, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto result = slice(tensor, Slice{0, 3, 2}, Slice{0, 4, 2});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 2);
    EXPECT_TRUE(result.size() == 4);

    EXPECT_TRUE(result(0, 0) == 1);
    EXPECT_TRUE(result(0, 1) == 3);
    EXPECT_TRUE(result(1, 0) == 9);
    EXPECT_TRUE(result(1, 1) == 11);
}

TEST(OpsSlice, tensor_rank_three_slice)
{
    Tensor<int> tensor(Shape{2, 3, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto result = slice(tensor, Slice{0, 2}, Slice{1, 3}, Slice{2, 4});

    EXPECT_TRUE(result.rank() == 3);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 2);
    EXPECT_TRUE(result.shape()(2) == 2);
    EXPECT_TRUE(result.size() == 8);

    EXPECT_TRUE(result(0, 0, 0) == tensor(0, 1, 2));
    EXPECT_TRUE(result(0, 0, 1) == tensor(0, 1, 3));
    EXPECT_TRUE(result(0, 1, 0) == tensor(0, 2, 2));
    EXPECT_TRUE(result(0, 1, 1) == tensor(0, 2, 3));
    EXPECT_TRUE(result(1, 0, 0) == tensor(1, 1, 2));
    EXPECT_TRUE(result(1, 0, 1) == tensor(1, 1, 3));
    EXPECT_TRUE(result(1, 1, 0) == tensor(1, 2, 2));
    EXPECT_TRUE(result(1, 1, 1) == tensor(1, 2, 3));
}

TEST(OpsSlice, tensor_empty_slice)
{
    Tensor<int> tensor(Shape{2, 3}, 7);

    auto result = slice(tensor, Slice{1, 1}, Slice{0, 3});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.shape()(0) == 0);
    EXPECT_TRUE(result.shape()(1) == 3);
    EXPECT_TRUE(result.size() == 0);
    EXPECT_TRUE(result.empty());
}

TEST(OpsSlice, tensor_zero_dimension_slice)
{
    Tensor<int> tensor(Shape{2, 0, 3});

    auto result = slice(tensor, Slice{0, 2}, Slice{0, 0}, Slice{0, 3});

    EXPECT_TRUE(result.rank() == 3);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 0);
    EXPECT_TRUE(result.shape()(2) == 3);
    EXPECT_TRUE(result.size() == 0);
    EXPECT_TRUE(result.empty());
}

TEST(OpsSlice, tensor_slice_rejects_rank_mismatch)
{
    Tensor<int> tensor(Shape{2, 3}, 1);

    bool threw = false;

    try {
        slice(tensor, Slice{0, 2});
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsSlice, tensor_slice_reversed_range_is_empty_with_positive_step)
{
    Tensor<int> tensor(Shape{2, 3}, 1);

    auto result = slice(tensor, Slice{1, 0}, Slice{0, 3});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.shape()(0) == 0);
    EXPECT_TRUE(result.shape()(1) == 3);
    EXPECT_TRUE(result.size() == 0);
    EXPECT_TRUE(result.empty());
}

TEST(OpsSlice, tensor_slice_supports_negative_step)
{
    Tensor<int> tensor(Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto result = slice(tensor, Slice{-1, -3, -1}, Slice{-1, -4, -2});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 2);
    EXPECT_TRUE(result.size() == 4);
    EXPECT_TRUE(result(0, 0) == 6);
    EXPECT_TRUE(result(0, 1) == 4);
    EXPECT_TRUE(result(1, 0) == 3);
    EXPECT_TRUE(result(1, 1) == 1);
}

TEST(OpsSlice, tensor_slice_clamps_out_of_bounds)
{
    Tensor<int> tensor(Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto result = slice(tensor, Slice{0, 2}, Slice{0, 4});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 3);
    EXPECT_TRUE(result.size() == 6);
    EXPECT_TRUE(result(0, 0) == 1);
    EXPECT_TRUE(result(0, 1) == 2);
    EXPECT_TRUE(result(0, 2) == 3);
    EXPECT_TRUE(result(1, 0) == 4);
    EXPECT_TRUE(result(1, 1) == 5);
    EXPECT_TRUE(result(1, 2) == 6);
}

