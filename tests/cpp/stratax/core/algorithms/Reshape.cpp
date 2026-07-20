#include <gtest/gtest.h>
#include <complex>
#include <limits>

#include <stratax.h>

using namespace stratax::container;
using stratax::core::Shape;

TEST(OpsReshape, vector_reshapes_to_tensor)
{
    Vector<int> vector{1, 2, 3, 4, 5, 6};

    auto tensor = reshape(vector, Shape{2, 3});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.size() == 6);

    EXPECT_TRUE(tensor[0] == 1);
    EXPECT_TRUE(tensor[1] == 2);
    EXPECT_TRUE(tensor[2] == 3);
    EXPECT_TRUE(tensor[3] == 4);
    EXPECT_TRUE(tensor[4] == 5);
    EXPECT_TRUE(tensor[5] == 6);
}

TEST(OpsReshape, matrix_reshapes_to_tensor)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    auto tensor = reshape(matrix, Shape{3, 2});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 3);
    EXPECT_TRUE(tensor.shape()(1) == 2);
    EXPECT_TRUE(tensor.size() == 6);

    EXPECT_TRUE(tensor(0, 0) == 1);
    EXPECT_TRUE(tensor(0, 1) == 2);
    EXPECT_TRUE(tensor(1, 0) == 3);
    EXPECT_TRUE(tensor(1, 1) == 4);
    EXPECT_TRUE(tensor(2, 0) == 5);
    EXPECT_TRUE(tensor(2, 1) == 6);
}

TEST(OpsReshape, tensor_reshapes_to_new_shape)
{
    Tensor<int> tensor(Shape{2, 2, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto reshaped = reshape(tensor, Shape{4, 2});

    EXPECT_TRUE(reshaped.rank() == 2);
    EXPECT_TRUE(reshaped.shape()(0) == 4);
    EXPECT_TRUE(reshaped.shape()(1) == 2);
    EXPECT_TRUE(reshaped.size() == 8);

    for (std::size_t i = 0; i < reshaped.size(); ++i) {
        EXPECT_TRUE(reshaped[i] == static_cast<int>(i + 1));
    }
}

TEST(OpsReshape, reshape_rejects_size_mismatch)
{
    Vector<int> vector{1, 2, 3, 4};

    bool threw = false;

    try {
        reshape(vector, Shape{3, 2});
    }
    catch (const Exceptions::ShapeError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsReshape, reshape_to_zero_sized_shape)
{
    Vector<int> vector(0);

    auto tensor = reshape(vector, Shape{0, 3});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 0);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.size() == 0);
    EXPECT_TRUE(tensor.empty());
}

TEST(OpsReshape, reshape_empty_vector_to_empty_shape)
{
    Vector<int> vector(0);

    auto tensor = reshape(vector, Shape{});

    EXPECT_TRUE(tensor.rank() == 0);
    EXPECT_TRUE(tensor.size() == 0);
    EXPECT_TRUE(tensor.empty());
    EXPECT_TRUE(tensor.shape().empty());
}

TEST(OpsReshape, reshape_zero_dimension_tensor)
{
    Tensor<int> tensor(Shape{2, 0, 3});

    auto reshaped = reshape(tensor, Shape{0});

    EXPECT_TRUE(reshaped.rank() == 1);
    EXPECT_TRUE(reshaped.shape()(0) == 0);
    EXPECT_TRUE(reshaped.size() == 0);
    EXPECT_TRUE(reshaped.empty());
}

TEST(OpsReshape, reshape_overflow_shape_throws)
{
    Vector<int> vector(0);

    EXPECT_THROW(
        reshape(vector, Shape{2, std::numeric_limits<long long>::max(), 2}),
        Exceptions::StrataxError);
}

TEST(OpsReshape, flatten_vector_copies_values)
{
    Vector<int> vector{1, 2, 3};

    auto flat = flatten(vector);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 3);
    EXPECT_TRUE(flat.size() == 3);
    EXPECT_TRUE(flat[0] == 1);
    EXPECT_TRUE(flat[1] == 2);
    EXPECT_TRUE(flat[2] == 3);
}

TEST(OpsReshape, flatten_matrix_copies_row_major_values)
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    auto flat = flatten(matrix);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 4);
    EXPECT_TRUE(flat.size() == 4);
    EXPECT_TRUE(flat[0] == 1);
    EXPECT_TRUE(flat[1] == 2);
    EXPECT_TRUE(flat[2] == 3);
    EXPECT_TRUE(flat[3] == 4);
}

TEST(OpsReshape, flatten_tensor_copies_linear_values)
{
    Tensor<int> tensor(Shape{2, 2});

    tensor[0] = 1;
    tensor[1] = 2;
    tensor[2] = 3;
    tensor[3] = 4;

    auto flat = flatten(tensor);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 4);
    EXPECT_TRUE(flat.size() == 4);
    EXPECT_TRUE(flat[0] == 1);
    EXPECT_TRUE(flat[1] == 2);
    EXPECT_TRUE(flat[2] == 3);
    EXPECT_TRUE(flat[3] == 4);
}

TEST(OpsReshape, flatten_empty_matrix_preserves_size)
{
    Matrix<int> matrix(0, 3);

    auto flat = flatten(matrix);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 0);
    EXPECT_TRUE(flat.size() == 0);
    EXPECT_TRUE(flat.empty());
}

TEST(OpsReshape, flatten_complex_tensor_copies_values)
{
    Tensor<std::complex<double>> tensor(Shape{2});

    tensor[0] = {1.0, 2.0};
    tensor[1] = {3.0, -4.0};

    auto flat = flatten(tensor);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 2);
    EXPECT_TRUE(flat[0] == std::complex<double>(1.0, 2.0));
    EXPECT_TRUE(flat[1] == std::complex<double>(3.0, -4.0));
}

TEST(OpsReshape, reshape_vector_to_3d)
{
    Vector<int> vector{1, 2, 3, 4, 5, 6, 7, 8};

    auto tensor = reshape(vector, Shape{2, 2, 2});

    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 2);
    EXPECT_TRUE(tensor.shape()(2) == 2);
    EXPECT_TRUE(tensor.size() == 8);
    EXPECT_TRUE(tensor[0] == 1);
    EXPECT_TRUE(tensor[7] == 8);
}

TEST(OpsReshape, reshape_vector_to_1d)
{
    Vector<int> vector{1, 2, 3};

    auto tensor = reshape(vector, Shape{3});

    EXPECT_TRUE(tensor.rank() == 1);
    EXPECT_TRUE(tensor.shape()(0) == 3);
    EXPECT_TRUE(tensor.size() == 3);
}

TEST(OpsReshape, reshape_vector_to_single_element)
{
    Vector<int> vector{42};

    auto tensor = reshape(vector, Shape{1, 1});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 1);
    EXPECT_TRUE(tensor.shape()(1) == 1);
    EXPECT_TRUE(tensor[0] == 42);
}

TEST(OpsReshape, reshape_matrix_to_1d)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    auto tensor = reshape(matrix, Shape{6});

    EXPECT_TRUE(tensor.rank() == 1);
    EXPECT_TRUE(tensor.shape()(0) == 6);
    EXPECT_TRUE(tensor.size() == 6);
    EXPECT_TRUE(tensor[0] == 1);
    EXPECT_TRUE(tensor[5] == 6);
}

TEST(OpsReshape, reshape_matrix_to_3d)
{
    Matrix<int> matrix{
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    };

    auto tensor = reshape(matrix, Shape{2, 2, 2});

    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 2);
    EXPECT_TRUE(tensor.shape()(2) == 2);
    EXPECT_TRUE(tensor.size() == 8);
}

TEST(OpsReshape, reshape_3d_to_2d)
{
    Tensor<int> tensor(Shape{2, 3, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto reshaped = reshape(tensor, Shape{6, 4});

    EXPECT_TRUE(reshaped.rank() == 2);
    EXPECT_TRUE(reshaped.shape()(0) == 6);
    EXPECT_TRUE(reshaped.shape()(1) == 4);
    EXPECT_TRUE(reshaped.size() == 24);
}

TEST(OpsReshape, reshape_4d_to_2d)
{
    Tensor<int> tensor(Shape{2, 2, 3, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto reshaped = reshape(tensor, Shape{4, 6});

    EXPECT_TRUE(reshaped.rank() == 2);
    EXPECT_TRUE(reshaped.shape()(0) == 4);
    EXPECT_TRUE(reshaped.shape()(1) == 6);
    EXPECT_TRUE(reshaped.size() == 24);
}

TEST(OpsReshape, reshape_float_vector)
{
    Vector<float> vector{1.5f, 2.5f, 3.5f, 4.5f};

    auto tensor = reshape(vector, Shape{2, 2});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 2);
    EXPECT_TRUE(tensor[0] == 1.5f);
    EXPECT_TRUE(tensor[3] == 4.5f);
}

TEST(OpsReshape, reshape_double_matrix)
{
    Matrix<double> matrix{
        {1.1, 2.2, 3.3},
        {4.4, 5.5, 6.6}
    };

    auto tensor = reshape(matrix, Shape{3, 2});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 3);
    EXPECT_TRUE(tensor.shape()(1) == 2);
    EXPECT_TRUE(tensor[0] > 1.0 && tensor[0] < 1.2);
}

TEST(OpsReshape, reshape_large_vector_to_matrix)
{
    Vector<int> vector(Shape{100});

    for (std::size_t i = 0; i < 100; ++i) {
        vector[i] = static_cast<int>(i);
    }

    auto matrix = reshape(vector, Shape{10, 10});

    EXPECT_TRUE(matrix.rank() == 2);
    EXPECT_TRUE(matrix.shape()(0) == 10);
    EXPECT_TRUE(matrix.shape()(1) == 10);
    EXPECT_TRUE(matrix.size() == 100);
    EXPECT_TRUE(matrix[0] == 0);
    EXPECT_TRUE(matrix[99] == 99);
}

TEST(OpsReshape, reshape_large_matrix_to_vector)
{
    Matrix<int> matrix(10, 10);

    int val = 0;
    for (std::size_t i = 0; i < 100; ++i) {
        matrix[i] = val++;
    }

    auto vector = reshape(matrix, Shape{100});

    EXPECT_TRUE(vector.rank() == 1);
    EXPECT_TRUE(vector.shape()(0) == 100);
    EXPECT_TRUE(vector.size() == 100);
    EXPECT_TRUE(vector[0] == 0);
    EXPECT_TRUE(vector[99] == 99);
}

TEST(OpsReshape, reshape_rectangular_to_different_rectangular)
{
    Matrix<int> matrix(3, 5);

    int val = 1;
    for (std::size_t i = 0; i < 15; ++i) {
        matrix[i] = val++;
    }

    auto reshaped = reshape(matrix, Shape{5, 3});

    EXPECT_TRUE(reshaped.rank() == 2);
    EXPECT_TRUE(reshaped.shape()(0) == 5);
    EXPECT_TRUE(reshaped.shape()(1) == 3);
    EXPECT_TRUE(reshaped.size() == 15);
    EXPECT_TRUE(reshaped[0] == 1);
    EXPECT_TRUE(reshaped[14] == 15);
}

TEST(OpsReshape, reshape_negative_values)
{
    Vector<int> vector{-1, -2, -3, -4};

    auto tensor = reshape(vector, Shape{2, 2});

    EXPECT_TRUE(tensor[0] == -1);
    EXPECT_TRUE(tensor[3] == -4);
}

TEST(OpsReshape, reshape_mixed_values)
{
    Vector<int> vector{-5, 0, 5, 10, -10, 100};

    auto tensor = reshape(vector, Shape{2, 3});

    EXPECT_TRUE(tensor[0] == -5);
    EXPECT_TRUE(tensor[1] == 0);
    EXPECT_TRUE(tensor[2] == 5);
}

TEST(OpsReshape, flatten_vector_to_vector)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    auto flat = flatten(vector);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 5);
    EXPECT_TRUE(flat[0] == 1);
    EXPECT_TRUE(flat[4] == 5);
}

TEST(OpsReshape, flatten_3d_tensor)
{
    Tensor<int> tensor(Shape{2, 3, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto flat = flatten(tensor);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 12);
    EXPECT_TRUE(flat.size() == 12);
    EXPECT_TRUE(flat[0] == 1);
    EXPECT_TRUE(flat[11] == 12);
}

TEST(OpsReshape, flatten_4d_tensor)
{
    Tensor<int> tensor(Shape{2, 2, 2, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto flat = flatten(tensor);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 16);
    EXPECT_TRUE(flat.size() == 16);
}

TEST(OpsReshape, flatten_float_matrix)
{
    Matrix<float> matrix{
        {1.5f, 2.5f, 3.5f},
        {4.5f, 5.5f, 6.5f}
    };

    auto flat = flatten(matrix);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 6);
    EXPECT_TRUE(flat[0] == 1.5f);
    EXPECT_TRUE(flat[5] == 6.5f);
}

TEST(OpsReshape, flatten_large_matrix)
{
    Matrix<int> matrix(50, 50);

    int val = 0;
    for (std::size_t i = 0; i < 2500; ++i) {
        matrix[i] = val++;
    }

    auto flat = flatten(matrix);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 2500);
    EXPECT_TRUE(flat.size() == 2500);
    EXPECT_TRUE(flat[0] == 0);
    EXPECT_TRUE(flat[2499] == 2499);
}

TEST(OpsReshape, flatten_single_row_matrix)
{
    Matrix<int> matrix(1, 5);

    for (std::size_t i = 0; i < 5; ++i) {
        matrix[i] = static_cast<int>(i + 1);
    }

    auto flat = flatten(matrix);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 5);
    EXPECT_TRUE(flat[0] == 1);
    EXPECT_TRUE(flat[4] == 5);
}

TEST(OpsReshape, flatten_single_column_matrix)
{
    Matrix<int> matrix(5, 1);

    for (std::size_t i = 0; i < 5; ++i) {
        matrix[i] = static_cast<int>(i + 10);
    }

    auto flat = flatten(matrix);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 5);
    EXPECT_TRUE(flat[0] == 10);
    EXPECT_TRUE(flat[4] == 14);
}

TEST(OpsReshape, flatten_negative_values)
{
    Matrix<int> matrix{
        {-1, -2},
        {-3, -4}
    };

    auto flat = flatten(matrix);

    EXPECT_TRUE(flat[0] == -1);
    EXPECT_TRUE(flat[3] == -4);
}

TEST(OpsReshape, reshape_rank_increase)
{
    Vector<int> vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    auto tensor = reshape(vector, Shape{2, 3, 2});

    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.shape()(2) == 2);
    EXPECT_TRUE(tensor.size() == 12);
}

TEST(OpsReshape, reshape_rank_decrease)
{
    Tensor<int> tensor(Shape{2, 3, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto flat = reshape(tensor, Shape{12});

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 12);
    EXPECT_TRUE(flat.size() == 12);
}

TEST(OpsReshape, reshape_complex_values)
{
    Vector<std::complex<double>> vector{{1, 2}, {3, 4}, {5, 6}, {7, 8}};

    auto tensor = reshape(vector, Shape{2, 2});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 2);
    EXPECT_TRUE(tensor[0] == std::complex<double>(1, 2));
    EXPECT_TRUE(tensor[3] == std::complex<double>(7, 8));
}

TEST(OpsReshape, flatten_asymmetric_tensor)
{
    Tensor<int> tensor(Shape{1, 5, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto flat = flatten(tensor);

    EXPECT_TRUE(flat.rank() == 1);
    EXPECT_TRUE(flat.shape()(0) == 15);
    EXPECT_TRUE(flat.size() == 15);
}

TEST(OpsReshape, reshape_to_rectangular_3x2)
{
    Vector<int> vector{1, 2, 3, 4, 5, 6};

    auto tensor = reshape(vector, Shape{3, 2});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 3);
    EXPECT_TRUE(tensor.shape()(1) == 2);
}

TEST(OpsReshape, reshape_to_rectangular_2x3)
{
    Vector<int> vector{1, 2, 3, 4, 5, 6};

    auto tensor = reshape(vector, Shape{2, 3});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 3);
}

