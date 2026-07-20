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

TEST(OpsSlice, vector_slice_single_element)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    auto result = slice(vector, Slice{2, 3});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 1);
    EXPECT_TRUE(result.size() == 1);
    EXPECT_TRUE(result[0] == 3);
}

TEST(OpsSlice, vector_slice_step_one)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    auto result = slice(vector, Slice{0, 5, 1});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 5);
    EXPECT_TRUE(result.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_TRUE(result[i] == static_cast<int>(i + 1));
    }
}

TEST(OpsSlice, vector_slice_from_middle)
{
    Vector<int> vector{10, 20, 30, 40, 50};

    auto result = slice(vector, Slice{1, 4});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result[0] == 20);
    EXPECT_TRUE(result[1] == 30);
    EXPECT_TRUE(result[2] == 40);
}

TEST(OpsSlice, vector_slice_to_end)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    auto result = slice(vector, Slice{2, 5});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result[0] == 3);
    EXPECT_TRUE(result[1] == 4);
    EXPECT_TRUE(result[2] == 5);
}

TEST(OpsSlice, vector_slice_large_step)
{
    Vector<int> vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto result = slice(vector, Slice{0, 10, 3});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 4);
    EXPECT_TRUE(result[0] == 1);
    EXPECT_TRUE(result[1] == 4);
    EXPECT_TRUE(result[2] == 7);
    EXPECT_TRUE(result[3] == 10);
}

TEST(OpsSlice, vector_slice_float)
{
    Vector<float> vector{1.5f, 2.5f, 3.5f, 4.5f, 5.5f};

    auto result = slice(vector, Slice{1, 4});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result[0] == 2.5f);
    EXPECT_TRUE(result[1] == 3.5f);
    EXPECT_TRUE(result[2] == 4.5f);
}

TEST(OpsSlice, vector_slice_double)
{
    Vector<double> vector{1.123, 2.456, 3.789, 4.012, 5.345};

    auto result = slice(vector, Slice{0, 3, 2});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result[0] == 1.123);
    EXPECT_TRUE(result[1] == 3.789);
}

TEST(OpsSlice, vector_slice_large_vector)
{
    Vector<int> vector(Shape{100});

    for (std::size_t i = 0; i < 100; ++i) {
        vector[i] = static_cast<int>(i);
    }

    auto result = slice(vector, Slice{10, 90, 10});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 8);
    EXPECT_TRUE(result[0] == 10);
    EXPECT_TRUE(result[7] == 80);
}

TEST(OpsSlice, vector_slice_negative_start_and_stop)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    auto result = slice(vector, Slice{-4, -1});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result[0] == 2);
    EXPECT_TRUE(result[1] == 3);
    EXPECT_TRUE(result[2] == 4);
}

TEST(OpsSlice, matrix_slice_single_element)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    auto result = slice(matrix, Slice{1, 2}, Slice{1, 2});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 1);
    EXPECT_TRUE(result.cols() == 1);
    EXPECT_TRUE(result(0, 0) == 5);
}

TEST(OpsSlice, matrix_slice_single_row)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    auto result = slice(matrix, Slice{1, 2}, Slice{0, 3});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 1);
    EXPECT_TRUE(result.cols() == 3);
    EXPECT_TRUE(result(0, 0) == 4);
    EXPECT_TRUE(result(0, 1) == 5);
    EXPECT_TRUE(result(0, 2) == 6);
}

TEST(OpsSlice, matrix_slice_single_column)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    auto result = slice(matrix, Slice{0, 3}, Slice{1, 2});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 3);
    EXPECT_TRUE(result.cols() == 1);
    EXPECT_TRUE(result(0, 0) == 2);
    EXPECT_TRUE(result(1, 0) == 5);
    EXPECT_TRUE(result(2, 0) == 8);
}

TEST(OpsSlice, matrix_slice_all_elements)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    auto result = slice(matrix, Slice{0, 2}, Slice{0, 3});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 2);
    EXPECT_TRUE(result.cols() == 3);
    EXPECT_TRUE(result(0, 0) == 1);
    EXPECT_TRUE(result(1, 2) == 6);
}

TEST(OpsSlice, matrix_slice_rectangular)
{
    Matrix<int> matrix(3, 5);

    int val = 1;
    for (std::size_t i = 0; i < 15; ++i) {
        matrix[i] = val++;
    }

    auto result = slice(matrix, Slice{0, 3}, Slice{1, 4});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 3);
    EXPECT_TRUE(result.cols() == 3);
    EXPECT_TRUE(result(0, 0) == 2);
    EXPECT_TRUE(result(2, 2) == 14);
}

TEST(OpsSlice, matrix_slice_float)
{
    Matrix<float> matrix{
        {1.1f, 2.2f, 3.3f},
        {4.4f, 5.5f, 6.6f}
    };

    auto result = slice(matrix, Slice{0, 2}, Slice{1, 3});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 2);
    EXPECT_TRUE(result.cols() == 2);
    EXPECT_TRUE(result(0, 0) == 2.2f);
    EXPECT_TRUE(result(1, 1) == 6.6f);
}

TEST(OpsSlice, matrix_slice_large_matrix)
{
    Matrix<int> matrix(50, 50);

    for (std::size_t i = 0; i < 2500; ++i) {
        matrix[i] = static_cast<int>(i);
    }

    auto result = slice(matrix, Slice{10, 30}, Slice{10, 30});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 20);
    EXPECT_TRUE(result.cols() == 20);
    EXPECT_TRUE(result(0, 0) == 510);
    EXPECT_TRUE(result(19, 19) == 1479);
}

TEST(OpsSlice, matrix_slice_step_2)
{
    Matrix<int> matrix{
        {1, 2, 3, 4, 5, 6},
        {7, 8, 9, 10, 11, 12}
    };

    auto result = slice(matrix, Slice{0, 2}, Slice{0, 6, 2});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 2);
    EXPECT_TRUE(result.cols() == 3);
    EXPECT_TRUE(result(0, 0) == 1);
    EXPECT_TRUE(result(0, 1) == 3);
    EXPECT_TRUE(result(0, 2) == 5);
}

TEST(OpsSlice, tensor_rank_4_slice)
{
    Tensor<int> tensor(Shape{2, 3, 2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto result = slice(tensor, Slice{0, 2}, Slice{0, 3}, Slice{0, 2}, Slice{0, 3});

    EXPECT_TRUE(result.rank() == 4);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 3);
    EXPECT_TRUE(result.shape()(2) == 2);
    EXPECT_TRUE(result.shape()(3) == 3);
    EXPECT_TRUE(result.size() == 36);
}

TEST(OpsSlice, tensor_rank_4_partial_slice)
{
    Tensor<int> tensor(Shape{3, 3, 3, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto result = slice(tensor, Slice{1, 3}, Slice{0, 3}, Slice{1, 3}, Slice{0, 3});

    EXPECT_TRUE(result.rank() == 4);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 3);
    EXPECT_TRUE(result.shape()(2) == 2);
    EXPECT_TRUE(result.shape()(3) == 3);
}

TEST(OpsSlice, tensor_rank_5_slice)
{
    Tensor<int> tensor(Shape{2, 2, 2, 2, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto result = slice(tensor, Slice{0, 2}, Slice{0, 2}, Slice{0, 2}, Slice{0, 2}, Slice{0, 2});

    EXPECT_TRUE(result.rank() == 5);
    EXPECT_TRUE(result.size() == 32);
}

TEST(OpsSlice, tensor_float_slice)
{
    Tensor<float> tensor(Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<float>(i) + 0.5f;
    }

    auto result = slice(tensor, Slice{0, 2}, Slice{1, 3});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 2);
    EXPECT_TRUE(result(0, 0) == 1.5f);
    EXPECT_TRUE(result(1, 1) == 5.5f);
}

TEST(OpsSlice, tensor_asymmetric_3d_slice)
{
    Tensor<int> tensor(Shape{2, 3, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto result = slice(tensor, Slice{0, 2}, Slice{0, 3}, Slice{0, 4});

    EXPECT_TRUE(result.rank() == 3);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 3);
    EXPECT_TRUE(result.shape()(2) == 4);
    EXPECT_TRUE(result.size() == 24);
}

TEST(OpsSlice, tensor_slice_with_all_steps)
{
    Tensor<int> tensor(Shape{4, 4, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto result = slice(tensor, Slice{0, 4, 2}, Slice{0, 4, 2}, Slice{0, 4, 2});

    EXPECT_TRUE(result.rank() == 3);
    EXPECT_TRUE(result.shape()(0) == 2);
    EXPECT_TRUE(result.shape()(1) == 2);
    EXPECT_TRUE(result.shape()(2) == 2);
    EXPECT_TRUE(result.size() == 8);
}

TEST(OpsSlice, vector_slice_start_equals_stop)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    auto result = slice(vector, Slice{2, 2});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 0);
    EXPECT_TRUE(result.size() == 0);
    EXPECT_TRUE(result.empty());
}

TEST(OpsSlice, matrix_slice_start_equals_stop_row)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    auto result = slice(matrix, Slice{1, 1}, Slice{0, 3});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 0);
    EXPECT_TRUE(result.cols() == 3);
    EXPECT_TRUE(result.empty());
}

TEST(OpsSlice, matrix_slice_start_equals_stop_col)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    auto result = slice(matrix, Slice{0, 2}, Slice{1, 1});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.rows() == 2);
    EXPECT_TRUE(result.cols() == 0);
    EXPECT_TRUE(result.empty());
}

TEST(OpsSlice, tensor_slice_negative_step_larger_tensor)
{
    Tensor<int> tensor(Shape{5, 5});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto result = slice(tensor, Slice{-1, -6, -2}, Slice{-1, -6, -2});

    EXPECT_TRUE(result.rank() == 2);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result.shape()(1) == 3);
    EXPECT_TRUE(result.size() == 9);
}

TEST(OpsSlice, vector_slice_negative_step_middle)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    auto result = slice(vector, Slice{-2, -5, -1});

    EXPECT_TRUE(result.rank() == 1);
    EXPECT_TRUE(result.shape()(0) == 3);
    EXPECT_TRUE(result[0] == 4);
    EXPECT_TRUE(result[1] == 3);
    EXPECT_TRUE(result[2] == 2);
}

