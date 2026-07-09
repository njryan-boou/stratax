#include <cassert>
#include <complex>

#include "stratax.hpp"

using namespace stratax::container;

void test_vector_equal()
{
    Vector<int> a{1, 2, 3};
    Vector<int> b{1, 2, 3};

    assert(a == b);
    assert(!(a != b));
}

void test_const_vector_equal()
{
    const Vector<int> a{1, 2, 3};
    const Vector<int> b{1, 2, 3};

    assert(a == b);
    assert(!(a != b));
}

void test_empty_vectors_equal()
{
    Vector<int> a(stratax::core::Shape{0});
    Vector<int> b(stratax::core::Shape{0});

    assert(a == b);
    assert(!(a != b));
}

void test_vector_not_equal_values()
{
    Vector<int> a{1, 2, 3};
    Vector<int> b{1, 9, 3};

    assert(!(a == b));
    assert(a != b);
}

void test_vector_not_equal_shape()
{
    Vector<int> a{1, 2, 3};
    Vector<int> b{1, 2};

    assert(!(a == b));
    assert(a != b);
}

void test_complex_vector_equal()
{
    Vector<std::complex<double>> a{
        {1.0, 2.0},
        {3.0, -4.0}
    };
    Vector<std::complex<double>> b{
        {1.0, 2.0},
        {3.0, -4.0}
    };

    assert(a == b);
    assert(!(a != b));

    b[1] = {3.0, 4.0};

    assert(!(a == b));
    assert(a != b);
}

void test_matrix_equal()
{
    Matrix<int> a{
        {1, 2},
        {3, 4}
    };
    Matrix<int> b{
        {1, 2},
        {3, 4}
    };

    assert(a == b);
    assert(!(a != b));
}

void test_matrix_not_equal_values()
{
    Matrix<int> a{
        {1, 2},
        {3, 4}
    };
    Matrix<int> b{
        {1, 2},
        {3, 9}
    };

    assert(!(a == b));
    assert(a != b);
}

void test_matrix_not_equal_shape()
{
    Matrix<int> a{
        {1, 2},
        {3, 4}
    };
    Matrix<int> b{
        {1, 2, 3},
        {4, 5, 6}
    };

    assert(!(a == b));
    assert(a != b);
}

void test_zero_dimension_matrices_equal()
{
    Matrix<int> a(0, 3);
    Matrix<int> b(0, 3);
    Matrix<int> c(0, 4);

    assert(a == b);
    assert(!(a != b));
    assert(!(a == c));
    assert(a != c);
}

void test_tensor_equal()
{
    Tensor<int> a(stratax::core::Shape{2, 2});
    Tensor<int> b(stratax::core::Shape{2, 2});

    a(0) = 1;
    a(1) = 2;
    a(2) = 3;
    a(3) = 4;

    b(0) = 1;
    b(1) = 2;
    b(2) = 3;
    b(3) = 4;

    assert(a == b);
    assert(!(a != b));
}

void test_tensor_not_equal_values()
{
    Tensor<int> a(stratax::core::Shape{2, 2}, 1);
    Tensor<int> b(stratax::core::Shape{2, 2}, 1);

    b(3) = 9;

    assert(!(a == b));
    assert(a != b);
}

void test_tensor_not_equal_shape()
{
    Tensor<int> a(stratax::core::Shape{2, 2}, 1);
    Tensor<int> b(stratax::core::Shape{4}, 1);

    assert(!(a == b));
    assert(a != b);
}

void test_empty_tensors_equal()
{
    Tensor<int> a(stratax::core::Shape{});
    Tensor<int> b(stratax::core::Shape{});

    assert(a == b);
    assert(!(a != b));
}

void test_zero_dimension_tensors_equal()
{
    Tensor<int> a(stratax::core::Shape{2, 0, 3});
    Tensor<int> b(stratax::core::Shape{2, 0, 3});
    Tensor<int> c(stratax::core::Shape{2, 0, 4});

    assert(a == b);
    assert(!(a != b));
    assert(!(a == c));
    assert(a != c);
}

int main()
{
    test_vector_equal();
    test_const_vector_equal();
    test_empty_vectors_equal();
    test_vector_not_equal_values();
    test_vector_not_equal_shape();
    test_complex_vector_equal();
    test_matrix_equal();
    test_matrix_not_equal_values();
    test_matrix_not_equal_shape();
    test_zero_dimension_matrices_equal();
    test_tensor_equal();
    test_tensor_not_equal_values();
    test_tensor_not_equal_shape();
    test_empty_tensors_equal();
    test_zero_dimension_tensors_equal();

    return 0;
}
