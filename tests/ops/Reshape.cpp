#include <cassert>
#include <complex>
#include <limits>

#include <stratax.hpp>

using namespace stratax::container;
using stratax::core::Shape;

void test_vector_reshapes_to_tensor()
{
    Vector<int> vector{1, 2, 3, 4, 5, 6};

    auto tensor = reshape(vector, Shape{2, 3});

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 2);
    assert(tensor.shape()(1) == 3);
    assert(tensor.size() == 6);

    assert(tensor[0] == 1);
    assert(tensor[1] == 2);
    assert(tensor[2] == 3);
    assert(tensor[3] == 4);
    assert(tensor[4] == 5);
    assert(tensor[5] == 6);
}

void test_matrix_reshapes_to_tensor()
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    auto tensor = reshape(matrix, Shape{3, 2});

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 3);
    assert(tensor.shape()(1) == 2);
    assert(tensor.size() == 6);

    assert(tensor(0, 0) == 1);
    assert(tensor(0, 1) == 2);
    assert(tensor(1, 0) == 3);
    assert(tensor(1, 1) == 4);
    assert(tensor(2, 0) == 5);
    assert(tensor(2, 1) == 6);
}

void test_tensor_reshapes_to_new_shape()
{
    Tensor<int> tensor(Shape{2, 2, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    auto reshaped = reshape(tensor, Shape{4, 2});

    assert(reshaped.rank() == 2);
    assert(reshaped.shape()(0) == 4);
    assert(reshaped.shape()(1) == 2);
    assert(reshaped.size() == 8);

    for (std::size_t i = 0; i < reshaped.size(); ++i) {
        assert(reshaped[i] == static_cast<int>(i + 1));
    }
}

void test_reshape_rejects_size_mismatch()
{
    Vector<int> vector{1, 2, 3, 4};

    bool threw = false;

    try {
        reshape(vector, Shape{3, 2});
    }
    catch (const Exceptions::ShapeError&) {
        threw = true;
    }

    assert(threw);
}

void test_reshape_to_zero_sized_shape()
{
    Vector<int> vector(0);

    auto tensor = reshape(vector, Shape{0, 3});

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 0);
    assert(tensor.shape()(1) == 3);
    assert(tensor.size() == 0);
    assert(tensor.empty());
}

void test_reshape_empty_vector_to_empty_shape()
{
    Vector<int> vector(0);

    auto tensor = reshape(vector, Shape{});

    assert(tensor.rank() == 0);
    assert(tensor.size() == 0);
    assert(tensor.empty());
    assert(tensor.shape().empty());
}

void test_reshape_zero_dimension_tensor()
{
    Tensor<int> tensor(Shape{2, 0, 3});

    auto reshaped = reshape(tensor, Shape{0});

    assert(reshaped.rank() == 1);
    assert(reshaped.shape()(0) == 0);
    assert(reshaped.size() == 0);
    assert(reshaped.empty());
}

void test_reshape_overflow_shape_throws()
{
    Vector<int> vector(0);

    bool threw = false;

    try {
        reshape(vector, Shape{2, std::numeric_limits<std::size_t>::max(), 2});
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    assert(threw);
}

void test_flatten_vector_copies_values()
{
    Vector<int> vector{1, 2, 3};

    auto flat = flatten(vector);

    assert(flat.rank() == 1);
    assert(flat.shape()(0) == 3);
    assert(flat.size() == 3);
    assert(flat[0] == 1);
    assert(flat[1] == 2);
    assert(flat[2] == 3);
}

void test_flatten_matrix_copies_row_major_values()
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    auto flat = flatten(matrix);

    assert(flat.rank() == 1);
    assert(flat.shape()(0) == 4);
    assert(flat.size() == 4);
    assert(flat[0] == 1);
    assert(flat[1] == 2);
    assert(flat[2] == 3);
    assert(flat[3] == 4);
}

void test_flatten_tensor_copies_linear_values()
{
    Tensor<int> tensor(Shape{2, 2});

    tensor[0] = 1;
    tensor[1] = 2;
    tensor[2] = 3;
    tensor[3] = 4;

    auto flat = flatten(tensor);

    assert(flat.rank() == 1);
    assert(flat.shape()(0) == 4);
    assert(flat.size() == 4);
    assert(flat[0] == 1);
    assert(flat[1] == 2);
    assert(flat[2] == 3);
    assert(flat[3] == 4);
}

void test_flatten_empty_matrix_preserves_size()
{
    Matrix<int> matrix(0, 3);

    auto flat = flatten(matrix);

    assert(flat.rank() == 1);
    assert(flat.shape()(0) == 0);
    assert(flat.size() == 0);
    assert(flat.empty());
}

void test_flatten_complex_tensor_copies_values()
{
    Tensor<std::complex<double>> tensor(Shape{2});

    tensor[0] = {1.0, 2.0};
    tensor[1] = {3.0, -4.0};

    auto flat = flatten(tensor);

    assert(flat.rank() == 1);
    assert(flat.shape()(0) == 2);
    assert(flat[0] == std::complex<double>(1.0, 2.0));
    assert(flat[1] == std::complex<double>(3.0, -4.0));
}

int main()
{
    test_vector_reshapes_to_tensor();
    test_matrix_reshapes_to_tensor();
    test_tensor_reshapes_to_new_shape();
    test_reshape_rejects_size_mismatch();
    test_reshape_to_zero_sized_shape();
    test_reshape_empty_vector_to_empty_shape();
    test_reshape_zero_dimension_tensor();
    test_reshape_overflow_shape_throws();
    test_flatten_vector_copies_values();
    test_flatten_matrix_copies_row_major_values();
    test_flatten_tensor_copies_linear_values();
    test_flatten_empty_matrix_preserves_size();
    test_flatten_complex_tensor_copies_values();

    return 0;
}
