#include <cassert>
#include <complex>
#include <sstream>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/io/Print.hpp>

using namespace stratax::container;
using namespace stratax::core;

void test_shape_prints_empty()
{
    Shape shape;

    std::ostringstream out;
    out << shape;

    assert(out.str() == "()");
}

void test_shape_prints_rank_one_with_trailing_comma()
{
    Shape shape{3};

    std::ostringstream out;
    out << shape;

    assert(out.str() == "(3,)");
}

void test_shape_prints_multiple_dimensions()
{
    Shape shape{2, 3, 4};

    std::ostringstream out;
    out << shape;

    assert(out.str() == "(2, 3, 4)");
}

void test_vector_prints_empty()
{
    Vector<int> vector;

    std::ostringstream out;
    out << vector;

    assert(out.str() == "[]");
}

void test_vector_prints_values()
{
    Vector<int> vector{1, 2, 3};

    std::ostringstream out;
    out << vector;

    assert(out.str() == "[1, 2, 3]");
}

void test_vector_prints_complex_values()
{
    Vector<std::complex<double>> vector{
        {1.0, 2.0},
        {3.0, -4.0}
    };

    std::ostringstream out;
    out << vector;

    assert(out.str() == "[(1,2), (3,-4)]");
}

void test_matrix_prints_empty_shape()
{
    Matrix<int> matrix(0, 0);

    std::ostringstream out;
    out << matrix;

    assert(out.str() == "[\n\n]");
}

void test_matrix_prints_zero_columns()
{
    Matrix<int> matrix(2, 0);

    std::ostringstream out;
    out << matrix;

    assert(out.str() == "[\n    []\n    []\n]");
}

void test_matrix_prints_values()
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    std::ostringstream out;
    out << matrix;

    assert(out.str() == "[\n    [1, 2]\n    [3, 4]\n]");
}

void test_tensor_prints_zero_length()
{
    Tensor<int> tensor(Shape{0});

    std::ostringstream out;
    out << tensor;

    assert(out.str() == "[]");
}

void test_tensor_prints_rank_one()
{
    Tensor<int> tensor(Shape{3});

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;

    std::ostringstream out;
    out << tensor;

    assert(out.str() == "[1, 2, 3]");
}

void test_tensor_prints_rank_two()
{
    Tensor<int> tensor(Shape{2, 2});

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;
    tensor(3) = 4;

    std::ostringstream out;
    out << tensor;

    assert(out.str() == "[\n    [1, 2],\n    [3, 4]\n]");
}

void test_tensor_prints_rank_three()
{
    Tensor<int> tensor(Shape{2, 2, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    std::ostringstream out;
    out << tensor;

    assert(out.str() ==
        "[\n"
        "    [\n"
        "        [1, 2],\n"
        "        [3, 4]\n"
        "    ],\n"
        "    [\n"
        "        [5, 6],\n"
        "        [7, 8]\n"
        "    ]\n"
        "]");
}

void test_tensor_prints_zero_dimension()
{
    Tensor<int> tensor(Shape{2, 0, 3});

    std::ostringstream out;
    out << tensor;

    assert(out.str() == "[]");
}

int main()
{
    test_shape_prints_empty();
    test_shape_prints_rank_one_with_trailing_comma();
    test_shape_prints_multiple_dimensions();
    test_vector_prints_empty();
    test_vector_prints_values();
    test_vector_prints_complex_values();
    test_matrix_prints_empty_shape();
    test_matrix_prints_zero_columns();
    test_matrix_prints_values();
    test_tensor_prints_zero_length();
    test_tensor_prints_rank_one();
    test_tensor_prints_rank_two();
    test_tensor_prints_rank_three();
    test_tensor_prints_zero_dimension();

    return 0;
}
