#include <cassert>
#include <stdexcept>

#include "stratax.hpp"

using namespace stratax::container;

void test_vector_elementwise_arithmetic()
{
    Vector<int> a{8, 12, 20};
    Vector<int> b{2, 3, 5};

    Vector<int> sum = a + b;
    Vector<int> diff = a - b;
    Vector<int> product = a * b;
    Vector<int> quotient = a / b;

    assert(sum(0) == 10);
    assert(sum(1) == 15);
    assert(sum(2) == 25);

    assert(diff(0) == 6);
    assert(diff(1) == 9);
    assert(diff(2) == 15);

    assert(product(0) == 16);
    assert(product(1) == 36);
    assert(product(2) == 100);

    assert(quotient(0) == 4);
    assert(quotient(1) == 4);
    assert(quotient(2) == 4);
}

void test_matrix_elementwise_arithmetic()
{
    Matrix<int> a{
        {8, 12},
        {20, 30}
    };
    Matrix<int> b{
        {2, 3},
        {5, 6}
    };

    Matrix<int> sum = a + b;
    Matrix<int> diff = a - b;
    Matrix<int> product = a * b;
    Matrix<int> quotient = a / b;

    assert(sum.rows() == 2);
    assert(sum.cols() == 2);
    assert(sum(0, 0) == 10);
    assert(sum(1, 1) == 36);

    assert(diff(0, 0) == 6);
    assert(diff(1, 1) == 24);

    assert(product(0, 1) == 36);
    assert(product(1, 0) == 100);

    assert(quotient(0, 0) == 4);
    assert(quotient(1, 1) == 5);
}

void test_tensor_elementwise_arithmetic()
{
    Tensor<int> a(stratax::core::Shape{2, 2});
    Tensor<int> b(stratax::core::Shape{2, 2});

    for (std::size_t i = 0; i < a.size(); ++i) {
        a(i) = static_cast<int>((i + 1) * 10);
        b(i) = static_cast<int>(i + 1);
    }

    Tensor<int> sum = a + b;
    Tensor<int> diff = a - b;
    Tensor<int> product = a * b;
    Tensor<int> quotient = a / b;

    assert(sum.rank() == 2);
    assert(sum.shape()(0) == 2);
    assert(sum.shape()(1) == 2);

    assert(sum(0) == 11);
    assert(diff(1) == 18);
    assert(product(2) == 90);
    assert(quotient(3) == 10);
}

void test_array_division_by_zero_throws()
{
    Vector<int> numerator{1, 2, 3};
    Vector<int> denominator{1, 0, 3};

    bool threw = false;

    try {
        numerator / denominator;
    }
    catch (const Exceptions::StrataxError&) {
        threw = true;
    }

    assert(threw);
}

void test_array_scalar_arithmetic()
{
    Vector<int> a{2, 4, 8};

    Vector<int> sum = a + 2;
    Vector<int> diff = a - 2;
    Vector<int> product = a * 2;
    Vector<int> quotient = a / 2;

    assert(sum(0) == 4);
    assert(sum(2) == 10);

    assert(diff(0) == 0);
    assert(diff(2) == 6);

    assert(product(0) == 4);
    assert(product(2) == 16);

    assert(quotient(0) == 1);
    assert(quotient(2) == 4);
}

void test_array_scalar_division_by_zero_throws()
{
    Vector<int> a{1, 2, 3};

    bool threw = false;

    try {
        a / 0;
    }
    catch (const Exceptions::StrataxError&) {
        threw = true;
    }

    assert(threw);
}

void test_scalar_array_arithmetic()
{
    Vector<int> a{2, 4, 8};

    Vector<int> sum = 2 + a;
    Vector<int> diff = 20 - a;
    Vector<int> product = 2 * a;
    Vector<int> quotient = 32 / a;

    assert(sum(0) == 4);
    assert(sum(2) == 10);

    assert(diff(0) == 18);
    assert(diff(2) == 12);

    assert(product(0) == 4);
    assert(product(2) == 16);

    assert(quotient(0) == 16);
    assert(quotient(1) == 8);
    assert(quotient(2) == 4);
}

void test_scalar_array_division_by_zero_throws()
{
    Vector<int> denominator{1, 0, 2};

    bool threw = false;

    try {
        10 / denominator;
    }
    catch (const Exceptions::StrataxError&) {
        threw = true;
    }

    assert(threw);
}

void test_compound_array_arithmetic()
{
    Vector<int> a{8, 12, 20};
    Vector<int> b{2, 3, 5};

    a += b;
    assert(a(0) == 10);
    assert(a(1) == 15);
    assert(a(2) == 25);

    a -= b;
    assert(a(0) == 8);
    assert(a(1) == 12);
    assert(a(2) == 20);

    a *= b;
    assert(a(0) == 16);
    assert(a(1) == 36);
    assert(a(2) == 100);

    a /= b;
    assert(a(0) == 8);
    assert(a(1) == 12);
    assert(a(2) == 20);
}

void test_compound_scalar_arithmetic()
{
    Vector<int> a{2, 4, 8};

    a += 2;
    assert(a(0) == 4);
    assert(a(2) == 10);

    a -= 2;
    assert(a(0) == 2);
    assert(a(2) == 8);

    a *= 2;
    assert(a(0) == 4);
    assert(a(2) == 16);

    a /= 2;
    assert(a(0) == 2);
    assert(a(2) == 8);
}

void test_unary_arithmetic()
{
    Vector<int> a{1, -2, 3};

    Vector<int> positive = +a;
    Vector<int> negative = -a;

    assert(positive(0) == 1);
    assert(positive(1) == -2);
    assert(positive(2) == 3);

    assert(negative(0) == -1);
    assert(negative(1) == 2);
    assert(negative(2) == -3);
}

int main()
{
    test_vector_elementwise_arithmetic();
    test_matrix_elementwise_arithmetic();
    test_tensor_elementwise_arithmetic();
    test_array_division_by_zero_throws();
    test_array_scalar_arithmetic();
    test_array_scalar_division_by_zero_throws();
    test_scalar_array_arithmetic();
    test_scalar_array_division_by_zero_throws();
    test_compound_array_arithmetic();
    test_compound_scalar_arithmetic();
    test_unary_arithmetic();

    return 0;
}
