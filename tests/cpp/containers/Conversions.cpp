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
    catch (const Exceptions::DimensionError&) {
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

TEST(ContainersConversions, to_matrix_rejects_non_rank_two)
{
    stratax::container::Vector<int> vector{1, 2, 3};

    bool threw = false;

    try {
        to_matrix(vector);
    }
    catch (const Exceptions::DimensionError&) {
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

