#include <cassert>
#include <complex>
#include <string>

#include <stratax/containers/Conversions.hpp>

template<typename To, typename From>
concept CanAstypeVector = requires(const stratax::container::Vector<From>& vector) {
    astype<To>(vector);
};

static_assert(CanAstypeVector<double, int>);
static_assert(!CanAstypeVector<std::string, int>);

void test_rank_one_tensor_converts_to_vector()
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{3});

    tensor[0] = 1;
    tensor[1] = 2;
    tensor[2] = 3;

    auto vector = to_vector(tensor);

    assert(vector.rank() == 1);
    assert(vector.shape()(0) == 3);
    assert(vector.size() == 3);
    assert(vector[0] == 1);
    assert(vector[1] == 2);
    assert(vector[2] == 3);
}

void test_to_vector_rejects_non_rank_one()
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{2, 2});

    bool threw = false;

    try {
        to_vector(tensor);
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_rank_one_tensor_converts_to_vector()
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{0});

    auto vector = to_vector(tensor);

    assert(vector.rank() == 1);
    assert(vector.shape()(0) == 0);
    assert(vector.size() == 0);
    assert(vector.empty());
}

void test_rank_two_tensor_converts_to_matrix()
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto matrix = to_matrix(tensor);

    assert(matrix.rank() == 2);
    assert(matrix.rows() == 2);
    assert(matrix.cols() == 3);
    assert(matrix.size() == 6);
    assert(matrix[0] == 1);
    assert(matrix[1] == 2);
    assert(matrix[2] == 3);
    assert(matrix[3] == 4);
    assert(matrix[4] == 5);
    assert(matrix[5] == 6);
}

void test_to_matrix_rejects_non_rank_two()
{
    stratax::container::Vector<int> vector{1, 2, 3};

    bool threw = false;

    try {
        to_matrix(vector);
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_rank_two_tensor_converts_to_matrix()
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{0, 3});

    auto matrix = to_matrix(tensor);

    assert(matrix.rank() == 2);
    assert(matrix.rows() == 0);
    assert(matrix.cols() == 3);
    assert(matrix.size() == 0);
    assert(matrix.empty());
}

void test_vector_converts_to_tensor()
{
    stratax::container::Vector<int> vector{4, 5, 6};

    auto tensor = to_tensor(vector);

    assert(tensor.rank() == 1);
    assert(tensor.shape()(0) == 3);
    assert(tensor.size() == 3);
    assert(tensor[0] == 4);
    assert(tensor[1] == 5);
    assert(tensor[2] == 6);
}

void test_matrix_converts_to_tensor()
{
    stratax::container::Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    auto tensor = to_tensor(matrix);

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 2);
    assert(tensor.shape()(1) == 2);
    assert(tensor.size() == 4);
    assert(tensor[0] == 1);
    assert(tensor[1] == 2);
    assert(tensor[2] == 3);
    assert(tensor[3] == 4);
}

void test_zero_dimension_matrix_converts_to_tensor()
{
    stratax::container::Matrix<int> matrix(0, 3);

    auto tensor = to_tensor(matrix);

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 0);
    assert(tensor.shape()(1) == 3);
    assert(tensor.size() == 0);
    assert(tensor.empty());
}

void test_vector_astype_converts_values()
{
    stratax::container::Vector<int> vector{1, 2, 3};

    auto converted = astype<double>(vector);

    assert(converted.rank() == 1);
    assert(converted.shape()(0) == 3);
    assert(converted[0] == 1.0);
    assert(converted[1] == 2.0);
    assert(converted[2] == 3.0);
}

void test_empty_vector_astype()
{
    stratax::container::Vector<int> vector(stratax::core::Shape{0});

    auto converted = astype<double>(vector);

    assert(converted.rank() == 1);
    assert(converted.shape()(0) == 0);
    assert(converted.size() == 0);
    assert(converted.empty());
}

void test_matrix_astype_converts_values()
{
    stratax::container::Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    auto converted = astype<double>(matrix);

    assert(converted.rank() == 2);
    assert(converted.rows() == 2);
    assert(converted.cols() == 2);
    assert(converted[0] == 1.0);
    assert(converted[1] == 2.0);
    assert(converted[2] == 3.0);
    assert(converted[3] == 4.0);
}

void test_tensor_astype_converts_values()
{
    stratax::container::Tensor<double> tensor(stratax::core::Shape{2, 2});

    tensor[0] = 1.9;
    tensor[1] = 2.1;
    tensor[2] = 3.8;
    tensor[3] = 4.2;

    auto converted = astype<int>(tensor);

    assert(converted.rank() == 2);
    assert(converted.shape()(0) == 2);
    assert(converted.shape()(1) == 2);
    assert(converted[0] == 1);
    assert(converted[1] == 2);
    assert(converted[2] == 3);
    assert(converted[3] == 4);
}

void test_tensor_astype_to_complex()
{
    stratax::container::Tensor<int> tensor(stratax::core::Shape{2}, 3);

    auto converted = astype<std::complex<double>>(tensor);

    assert(converted.rank() == 1);
    assert(converted.shape()(0) == 2);
    assert(converted[0] == std::complex<double>{3});
    assert(converted[1] == std::complex<double>{3});
}

int main()
{
    test_rank_one_tensor_converts_to_vector();
    test_to_vector_rejects_non_rank_one();
    test_empty_rank_one_tensor_converts_to_vector();
    test_rank_two_tensor_converts_to_matrix();
    test_to_matrix_rejects_non_rank_two();
    test_empty_rank_two_tensor_converts_to_matrix();
    test_vector_converts_to_tensor();
    test_matrix_converts_to_tensor();
    test_zero_dimension_matrix_converts_to_tensor();
    test_vector_astype_converts_values();
    test_empty_vector_astype();
    test_matrix_astype_converts_values();
    test_tensor_astype_converts_values();
    test_tensor_astype_to_complex();

    return 0;
}
