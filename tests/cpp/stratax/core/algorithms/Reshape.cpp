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

