#include <gtest/gtest.h>
#include <complex>
#include <string>

#include <stratax.h>

template<typename To, typename From>
concept CanAstypeVector = requires(const stratax::container::Vector<From>& vector) {
    astype<To>(vector);
};

static_assert(CanAstypeVector<double, int>);
static_assert(!CanAstypeVector<std::string, int>);

TEST(ContainersConversions, rank_one_tensor_converts_to_vector)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{3});

    tensor[0] = 1;
    tensor[1] = 2;
    tensor[2] = 3;

    auto vector = to_vector(tensor);

    EXPECT_TRUE(vector.rank() == 1);
    EXPECT_TRUE(vector.shape()(0) == 3);
    EXPECT_TRUE(vector.size() == 3);
    EXPECT_TRUE(vector[0] == 1);
    EXPECT_TRUE(vector[1] == 2);
    EXPECT_TRUE(vector[2] == 3);
}

TEST(ContainersConversions, to_vector_rejects_non_rank_one)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{2, 2});

    bool threw = false;

    try {
        to_vector(tensor);
    }
    catch (const Exceptions::ShapeError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersConversions, empty_rank_one_tensor_converts_to_vector)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{0});

    auto vector = to_vector(tensor);

    EXPECT_TRUE(vector.rank() == 1);
    EXPECT_TRUE(vector.shape()(0) == 0);
    EXPECT_TRUE(vector.size() == 0);
    EXPECT_TRUE(vector.empty());
}

TEST(ContainersConversions, rank_two_tensor_converts_to_matrix)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto matrix = to_matrix(tensor);

    EXPECT_TRUE(matrix.rank() == 2);
    EXPECT_TRUE(matrix.rows() == 2);
    EXPECT_TRUE(matrix.cols() == 3);
    EXPECT_TRUE(matrix.size() == 6);
    EXPECT_TRUE(matrix[0] == 1);
    EXPECT_TRUE(matrix[1] == 2);
    EXPECT_TRUE(matrix[2] == 3);
    EXPECT_TRUE(matrix[3] == 4);
    EXPECT_TRUE(matrix[4] == 5);
    EXPECT_TRUE(matrix[5] == 6);
}

TEST(ContainersConversions, matrix_shaped_tensor_with_singletons_converts_to_matrix)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{1, 2, 1, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto matrix = to_matrix(tensor);

    EXPECT_TRUE(matrix.rank() == 2);
    EXPECT_TRUE(matrix.rows() == 2);
    EXPECT_TRUE(matrix.cols() == 3);
    EXPECT_TRUE(matrix.size() == 6);
    EXPECT_TRUE(matrix[0] == 1);
    EXPECT_TRUE(matrix[1] == 2);
    EXPECT_TRUE(matrix[2] == 3);
    EXPECT_TRUE(matrix[3] == 4);
    EXPECT_TRUE(matrix[4] == 5);
    EXPECT_TRUE(matrix[5] == 6);
}

TEST(ContainersConversions, to_matrix_rejects_non_rank_two)
{
    stratax::container::Vector<int> vector{1, 2, 3};

    bool threw = false;

    try {
        to_matrix(vector);
    }
    catch (const Exceptions::ShapeError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersConversions, to_matrix_rejects_non_matrix_shape)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{3});

    bool threw = false;

    try {
        to_matrix(tensor);
    }
    catch (const Exceptions::ShapeError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersConversions, empty_rank_two_tensor_converts_to_matrix)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{0, 3});

    auto matrix = to_matrix(tensor);

    EXPECT_TRUE(matrix.rank() == 2);
    EXPECT_TRUE(matrix.rows() == 0);
    EXPECT_TRUE(matrix.cols() == 3);
    EXPECT_TRUE(matrix.size() == 0);
    EXPECT_TRUE(matrix.empty());
}

TEST(ContainersConversions, vector_converts_to_tensor)
{
    stratax::container::Vector<int> vector{4, 5, 6};

    auto tensor = to_tensor(vector);

    EXPECT_TRUE(tensor.rank() == 1);
    EXPECT_TRUE(tensor.shape()(0) == 3);
    EXPECT_TRUE(tensor.size() == 3);
    EXPECT_TRUE(tensor[0] == 4);
    EXPECT_TRUE(tensor[1] == 5);
    EXPECT_TRUE(tensor[2] == 6);
}

TEST(ContainersConversions, matrix_converts_to_tensor)
{
    stratax::container::Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    auto tensor = to_tensor(matrix);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 2);
    EXPECT_TRUE(tensor.size() == 4);
    EXPECT_TRUE(tensor[0] == 1);
    EXPECT_TRUE(tensor[1] == 2);
    EXPECT_TRUE(tensor[2] == 3);
    EXPECT_TRUE(tensor[3] == 4);
}

TEST(ContainersConversions, zero_dimension_matrix_converts_to_tensor)
{
    stratax::container::Matrix<int> matrix(0, 3);

    auto tensor = to_tensor(matrix);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 0);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.size() == 0);
    EXPECT_TRUE(tensor.empty());
}

TEST(ContainersConversions, vector_astype_converts_values)
{
    stratax::container::Vector<int> vector{1, 2, 3};

    auto converted = astype<double>(vector);

    EXPECT_TRUE(converted.rank() == 1);
    EXPECT_TRUE(converted.shape()(0) == 3);
    EXPECT_TRUE(converted[0] == 1.0);
    EXPECT_TRUE(converted[1] == 2.0);
    EXPECT_TRUE(converted[2] == 3.0);
}

TEST(ContainersConversions, empty_vector_astype)
{
    stratax::container::Vector<int> vector(stratax::core::Shape{0});

    auto converted = astype<double>(vector);

    EXPECT_TRUE(converted.rank() == 1);
    EXPECT_TRUE(converted.shape()(0) == 0);
    EXPECT_TRUE(converted.size() == 0);
    EXPECT_TRUE(converted.empty());
}

TEST(ContainersConversions, matrix_astype_converts_values)
{
    stratax::container::Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    auto converted = astype<double>(matrix);

    EXPECT_TRUE(converted.rank() == 2);
    EXPECT_TRUE(converted.rows() == 2);
    EXPECT_TRUE(converted.cols() == 2);
    EXPECT_TRUE(converted[0] == 1.0);
    EXPECT_TRUE(converted[1] == 2.0);
    EXPECT_TRUE(converted[2] == 3.0);
    EXPECT_TRUE(converted[3] == 4.0);
}

TEST(ContainersConversions, tensor_astype_converts_values)
{
    stratax::container::Tensor<double> tensor(stratax::core::Shape{2, 2});

    tensor[0] = 1.9;
    tensor[1] = 2.1;
    tensor[2] = 3.8;
    tensor[3] = 4.2;

    auto converted = astype<int>(tensor);

    EXPECT_TRUE(converted.rank() == 2);
    EXPECT_TRUE(converted.shape()(0) == 2);
    EXPECT_TRUE(converted.shape()(1) == 2);
    EXPECT_TRUE(converted[0] == 1);
    EXPECT_TRUE(converted[1] == 2);
    EXPECT_TRUE(converted[2] == 3);
    EXPECT_TRUE(converted[3] == 4);
}

TEST(ContainersConversions, tensor_astype_to_complex)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{2}, 3);

    auto converted = astype<std::complex<double>>(tensor);

    EXPECT_TRUE(converted.rank() == 1);
    EXPECT_TRUE(converted.shape()(0) == 2);
    EXPECT_TRUE(converted[0] == std::complex<double>{3});
    EXPECT_TRUE(converted[1] == std::complex<double>{3});
}

TEST(ContainersConversions, vector_astype_int_to_float)
{
    stratax::container::Vector<int> vector{1, 2, 3, 4, 5};

    auto converted = astype<float>(vector);

    EXPECT_TRUE(converted.rank() == 1);
    EXPECT_TRUE(converted.shape()(0) == 5);
    EXPECT_TRUE(converted[0] == 1.0f);
    EXPECT_TRUE(converted[4] == 5.0f);
}

TEST(ContainersConversions, vector_astype_float_to_int)
{
    stratax::container::Vector<float> vector{1.9f, 2.1f, 3.5f, 4.2f};

    auto converted = astype<int>(vector);

    EXPECT_TRUE(converted.rank() == 1);
    EXPECT_TRUE(converted[0] == 1);
    EXPECT_TRUE(converted[1] == 2);
    EXPECT_TRUE(converted[2] == 3);
    EXPECT_TRUE(converted[3] == 4);
}

TEST(ContainersConversions, vector_astype_double_to_int)
{
    stratax::container::Vector<double> vector{1.999, 2.001, 3.5};

    auto converted = astype<int>(vector);

    EXPECT_TRUE(converted[0] == 1);
    EXPECT_TRUE(converted[1] == 2);
    EXPECT_TRUE(converted[2] == 3);
}

TEST(ContainersConversions, vector_astype_negative_values)
{
    stratax::container::Vector<int> vector{-1, -2, -3};

    auto converted = astype<double>(vector);

    EXPECT_TRUE(converted[0] == -1.0);
    EXPECT_TRUE(converted[1] == -2.0);
    EXPECT_TRUE(converted[2] == -3.0);
}

TEST(ContainersConversions, vector_astype_zero_values)
{
    stratax::container::Vector<int> vector{0, 0, 0};

    auto converted = astype<double>(vector);

    EXPECT_TRUE(converted[0] == 0.0);
    EXPECT_TRUE(converted[1] == 0.0);
    EXPECT_TRUE(converted[2] == 0.0);
}

TEST(ContainersConversions, matrix_astype_int_to_double)
{
    stratax::container::Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    auto converted = astype<double>(matrix);

    EXPECT_TRUE(converted.rows() == 2);
    EXPECT_TRUE(converted.cols() == 3);
    EXPECT_TRUE(converted[0] == 1.0);
    EXPECT_TRUE(converted[5] == 6.0);
}

TEST(ContainersConversions, matrix_astype_float_to_int)
{
    stratax::container::Matrix<float> matrix{
        {1.5f, 2.5f},
        {3.5f, 4.5f}
    };

    auto converted = astype<int>(matrix);

    EXPECT_TRUE(converted.rows() == 2);
    EXPECT_TRUE(converted.cols() == 2);
    EXPECT_TRUE(converted[0] == 1);
    EXPECT_TRUE(converted[1] == 2);
    EXPECT_TRUE(converted[2] == 3);
    EXPECT_TRUE(converted[3] == 4);
}

TEST(ContainersConversions, matrix_astype_negative_values)
{
    stratax::container::Matrix<int> matrix{
        {-1, -2},
        {-3, -4}
    };

    auto converted = astype<double>(matrix);

    EXPECT_TRUE(converted[0] == -1.0);
    EXPECT_TRUE(converted[3] == -4.0);
}

TEST(ContainersConversions, tensor_astype_int_to_unsigned)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{3}, 5);

    auto converted = astype<unsigned int>(tensor);

    EXPECT_TRUE(converted.shape()(0) == 3);
    EXPECT_TRUE(converted[0] == 5u);
    EXPECT_TRUE(converted[2] == 5u);
}

TEST(ContainersConversions, tensor_astype_float_to_double)
{
    stratax::container::Tensor<float> tensor(stratax::core::Shape{2, 2});

    tensor[0] = 1.5f;
    tensor[1] = 2.5f;
    tensor[2] = 3.5f;
    tensor[3] = 4.5f;

    auto converted = astype<double>(tensor);

    EXPECT_TRUE(converted[0] > 1.4 && converted[0] < 1.6);
    EXPECT_TRUE(converted[1] > 2.4 && converted[1] < 2.6);
}

TEST(ContainersConversions, tensor_rank_3_astype)
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{2, 2, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i);
    }

    auto converted = astype<double>(tensor);

    EXPECT_TRUE(converted.rank() == 3);
    EXPECT_TRUE(converted.shape()(0) == 2);
    EXPECT_TRUE(converted.shape()(1) == 2);
    EXPECT_TRUE(converted.shape()(2) == 2);
    EXPECT_TRUE(converted[0] == 0.0);
    EXPECT_TRUE(converted[7] == 7.0);
}

TEST(ContainersConversions, large_vector_astype)
{
    stratax::container::Vector<int> vector(stratax::core::Shape{100});

    for (std::size_t i = 0; i < 100; ++i) {
        vector[i] = static_cast<int>(i);
    }

    auto converted = astype<double>(vector);

    EXPECT_TRUE(converted.shape()(0) == 100);
    EXPECT_TRUE(converted[0] == 0.0);
    EXPECT_TRUE(converted[99] == 99.0);
}

TEST(ContainersConversions, large_matrix_astype)
{
    stratax::container::Matrix<int> matrix(10, 10);

    int val = 0;
    for (std::size_t i = 0; i < 100; ++i) {
        matrix[i] = val++;
    }

    auto converted = astype<float>(matrix);

    EXPECT_TRUE(converted.rows() == 10);
    EXPECT_TRUE(converted.cols() == 10);
    EXPECT_TRUE(converted[0] == 0.0f);
    EXPECT_TRUE(converted[99] == 99.0f);
}

TEST(ContainersConversions, single_element_vector_astype)
{
    stratax::container::Vector<int> vector{42};

    auto converted = astype<double>(vector);

    EXPECT_TRUE(converted.shape()(0) == 1);
    EXPECT_TRUE(converted[0] == 42.0);
}

TEST(ContainersConversions, single_element_matrix_astype)
{
    stratax::container::Matrix<int> matrix(1, 1);
    matrix[0] = 99;

    auto converted = astype<double>(matrix);

    EXPECT_TRUE(converted.rows() == 1);
    EXPECT_TRUE(converted.cols() == 1);
    EXPECT_TRUE(converted[0] == 99.0);
}

TEST(ContainersConversions, rectangular_matrix_astype)
{
    stratax::container::Matrix<int> matrix(3, 5);

    int val = 1;
    for (std::size_t i = 0; i < 15; ++i) {
        matrix[i] = val++;
    }

    auto converted = astype<double>(matrix);

    EXPECT_TRUE(converted.rows() == 3);
    EXPECT_TRUE(converted.cols() == 5);
    EXPECT_TRUE(converted[0] == 1.0);
    EXPECT_TRUE(converted[14] == 15.0);
}

TEST(ContainersConversions, vector_astype_to_complex_from_int)
{
    stratax::container::Vector<int> vector{1, 2, 3};

    auto converted = astype<std::complex<double>>(vector);

    EXPECT_TRUE(converted.shape()(0) == 3);
    EXPECT_TRUE(converted[0] == std::complex<double>(1, 0));
    EXPECT_TRUE(converted[2] == std::complex<double>(3, 0));
}

TEST(ContainersConversions, matrix_astype_to_complex_from_int)
{
    stratax::container::Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    auto converted = astype<std::complex<double>>(matrix);

    EXPECT_TRUE(converted.rows() == 2);
    EXPECT_TRUE(converted.cols() == 2);
    EXPECT_TRUE(converted[0] == std::complex<double>(1, 0));
    EXPECT_TRUE(converted[3] == std::complex<double>(4, 0));
}

TEST(ContainersConversions, vector_astype_to_long)
{
    stratax::container::Vector<int> vector{100, 200, 300};

    auto converted = astype<long>(vector);

    EXPECT_TRUE(converted[0] == 100L);
    EXPECT_TRUE(converted[1] == 200L);
    EXPECT_TRUE(converted[2] == 300L);
}

TEST(ContainersConversions, zero_dimension_vector_astype)
{
    stratax::container::Vector<int> vector(stratax::core::Shape{0});

    auto converted = astype<double>(vector);

    EXPECT_TRUE(converted.shape()(0) == 0);
    EXPECT_TRUE(converted.empty());
}

TEST(ContainersConversions, zero_dimension_matrix_astype)
{
    stratax::container::Matrix<int> matrix(0, 5);

    auto converted = astype<double>(matrix);

    EXPECT_TRUE(converted.rows() == 0);
    EXPECT_TRUE(converted.cols() == 5);
    EXPECT_TRUE(converted.empty());
}

TEST(ContainersConversions, tensor_converts_to_vector_rank_one)
{
    stratax::container::Tensor<double> tensor(stratax::core::Shape{5});

    for (std::size_t i = 0; i < 5; ++i) {
        tensor[i] = static_cast<double>(i) * 1.5;
    }

    auto vector = to_vector(tensor);

    EXPECT_TRUE(vector.rank() == 1);
    EXPECT_TRUE(vector.shape()(0) == 5);
    EXPECT_TRUE(vector[0] == 0.0);
    EXPECT_TRUE(vector[4] > 5.9 && vector[4] < 6.1);
}

TEST(ContainersConversions, tensor_converts_to_matrix_rank_two)
{
    stratax::container::Tensor<double> tensor(stratax::core::Shape{3, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<double>(i);
    }

    auto matrix = to_matrix(tensor);

    EXPECT_TRUE(matrix.rank() == 2);
    EXPECT_TRUE(matrix.rows() == 3);
    EXPECT_TRUE(matrix.cols() == 4);
    EXPECT_TRUE(matrix[0] == 0.0);
    EXPECT_TRUE(matrix[11] == 11.0);
}

TEST(ContainersConversions, astype_idempotent_same_type)
{
    stratax::container::Vector<int> vector{1, 2, 3};

    auto converted = astype<int>(vector);

    EXPECT_TRUE(converted.shape()(0) == 3);
    EXPECT_TRUE(converted[0] == 1);
    EXPECT_TRUE(converted[2] == 3);
}

TEST(ContainersConversions, matrix_mixed_values_astype)
{
    stratax::container::Matrix<int> matrix{
        {-5, 0, 5},
        {10, -10, 100}
    };

    auto converted = astype<double>(matrix);

    EXPECT_TRUE(converted[0] == -5.0);
    EXPECT_TRUE(converted[1] == 0.0);
    EXPECT_TRUE(converted[2] == 5.0);
    EXPECT_TRUE(converted[5] == 100.0);
}

