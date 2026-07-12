#include <gtest/gtest.h>
#include <complex>
#include <stdexcept>

#include <stratax.h>

using namespace stratax::container;

TEST(OpsArithmetic, vector_elementwise_arithmetic)
{
    Vector<int> a{8, 12, 20};
    Vector<int> b{2, 3, 5};

    Vector<int> sum = a + b;
    Vector<int> diff = a - b;
    Vector<int> product = a * b;
    Vector<int> quotient = a / b;

    EXPECT_TRUE(sum(0) == 10);
    EXPECT_TRUE(sum(1) == 15);
    EXPECT_TRUE(sum(2) == 25);

    EXPECT_TRUE(diff(0) == 6);
    EXPECT_TRUE(diff(1) == 9);
    EXPECT_TRUE(diff(2) == 15);

    EXPECT_TRUE(product(0) == 16);
    EXPECT_TRUE(product(1) == 36);
    EXPECT_TRUE(product(2) == 100);

    EXPECT_TRUE(quotient(0) == 4);
    EXPECT_TRUE(quotient(1) == 4);
    EXPECT_TRUE(quotient(2) == 4);
}

TEST(OpsArithmetic, matrix_elementwise_arithmetic)
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

    EXPECT_TRUE(sum.rows() == 2);
    EXPECT_TRUE(sum.cols() == 2);
    EXPECT_TRUE(sum(0, 0) == 10);
    EXPECT_TRUE(sum(1, 1) == 36);

    EXPECT_TRUE(diff(0, 0) == 6);
    EXPECT_TRUE(diff(1, 1) == 24);

    EXPECT_TRUE(product(0, 1) == 36);
    EXPECT_TRUE(product(1, 0) == 100);

    EXPECT_TRUE(quotient(0, 0) == 4);
    EXPECT_TRUE(quotient(1, 1) == 5);
}

TEST(OpsArithmetic, tensor_elementwise_arithmetic)
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

    EXPECT_TRUE(sum.rank() == 2);
    EXPECT_TRUE(sum.shape()(0) == 2);
    EXPECT_TRUE(sum.shape()(1) == 2);

    EXPECT_TRUE(sum(0) == 11);
    EXPECT_TRUE(diff(1) == 18);
    EXPECT_TRUE(product(2) == 90);
    EXPECT_TRUE(quotient(3) == 10);
}

TEST(OpsArithmetic, array_division_by_zero_throws)
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

    EXPECT_TRUE(threw);
}

TEST(OpsArithmetic, complex_array_division_by_zero_throws)
{
    Vector<std::complex<double>> numerator{
        {1.0, 0.0},
        {2.0, 0.0}
    };
    Vector<std::complex<double>> denominator{
        {1.0, 0.0},
        {}
    };

    bool threw = false;

    try {
        numerator / denominator;
    }
    catch (const Exceptions::ZeroDivisionError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsArithmetic, array_shape_mismatch_throws)
{
    Vector<int> short_vector{1, 2};
    Vector<int> long_vector{1, 2, 3};

    bool vector_threw = false;

    try {
        short_vector + long_vector;
    }
    catch (const Exceptions::ShapeError&) {
        vector_threw = true;
    }

    EXPECT_TRUE(vector_threw);

    Matrix<int> wide{
        {1, 2, 3},
        {4, 5, 6}
    };
    Matrix<int> tall{
        {1, 2},
        {3, 4},
        {5, 6}
    };

    bool matrix_threw = false;

    try {
        wide - tall;
    }
    catch (const Exceptions::ShapeError&) {
        matrix_threw = true;
    }

    EXPECT_TRUE(matrix_threw);

    Tensor<int> flat(stratax::core::Shape{4}, 1);
    Tensor<int> square(stratax::core::Shape{2, 2}, 1);

    bool tensor_threw = false;

    try {
        flat * square;
    }
    catch (const Exceptions::ShapeError&) {
        tensor_threw = true;
    }

    EXPECT_TRUE(tensor_threw);
}

TEST(OpsArithmetic, compound_array_shape_mismatch_throws)
{
    Vector<int> lhs{1, 2};
    Vector<int> rhs{1, 2, 3};

    bool threw = false;

    try {
        lhs += rhs;
    }
    catch (const Exceptions::ShapeError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
    EXPECT_TRUE(lhs.size() == 2);
    EXPECT_TRUE(lhs[0] == 1);
    EXPECT_TRUE(lhs[1] == 2);
}

TEST(OpsArithmetic, array_scalar_arithmetic)
{
    Vector<int> a{2, 4, 8};

    Vector<int> sum = a + 2;
    Vector<int> diff = a - 2;
    Vector<int> product = a * 2;
    Vector<int> quotient = a / 2;

    EXPECT_TRUE(sum(0) == 4);
    EXPECT_TRUE(sum(2) == 10);

    EXPECT_TRUE(diff(0) == 0);
    EXPECT_TRUE(diff(2) == 6);

    EXPECT_TRUE(product(0) == 4);
    EXPECT_TRUE(product(2) == 16);

    EXPECT_TRUE(quotient(0) == 1);
    EXPECT_TRUE(quotient(2) == 4);
}

TEST(OpsArithmetic, complex_array_scalar_arithmetic)
{
    Vector<std::complex<double>> a{
        {2.0, 1.0},
        {4.0, -2.0}
    };

    const std::complex<double> scalar{1.0, 2.0};

    auto sum = a + scalar;
    auto diff = a - scalar;
    auto product = a * scalar;
    auto quotient = a / std::complex<double>{2.0, 0.0};

    EXPECT_TRUE(sum[0] == std::complex<double>(3.0, 3.0));
    EXPECT_TRUE(diff[1] == std::complex<double>(3.0, -4.0));
    EXPECT_TRUE(product[0] == std::complex<double>(0.0, 5.0));
    EXPECT_TRUE(quotient[1] == std::complex<double>(2.0, -1.0));
}

TEST(OpsArithmetic, array_scalar_division_by_zero_throws)
{
    Vector<int> a{1, 2, 3};

    bool threw = false;

    try {
        a / 0;
    }
    catch (const Exceptions::StrataxError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsArithmetic, complex_scalar_division_by_zero_throws)
{
    Vector<std::complex<double>> a{
        {1.0, 0.0},
        {2.0, 0.0}
    };

    bool threw = false;

    try {
        a / std::complex<double>{};
    }
    catch (const Exceptions::ZeroDivisionError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsArithmetic, scalar_array_arithmetic)
{
    Vector<int> a{2, 4, 8};

    Vector<int> sum = 2 + a;
    Vector<int> diff = 20 - a;
    Vector<int> product = 2 * a;
    Vector<int> quotient = 32 / a;

    EXPECT_TRUE(sum(0) == 4);
    EXPECT_TRUE(sum(2) == 10);

    EXPECT_TRUE(diff(0) == 18);
    EXPECT_TRUE(diff(2) == 12);

    EXPECT_TRUE(product(0) == 4);
    EXPECT_TRUE(product(2) == 16);

    EXPECT_TRUE(quotient(0) == 16);
    EXPECT_TRUE(quotient(1) == 8);
    EXPECT_TRUE(quotient(2) == 4);
}

TEST(OpsArithmetic, complex_scalar_array_arithmetic)
{
    Vector<std::complex<double>> a{
        {2.0, 1.0},
        {4.0, -2.0}
    };

    const std::complex<double> scalar{10.0, 0.0};

    auto sum = scalar + a;
    auto diff = scalar - a;
    auto product = scalar * a;
    auto quotient = scalar / a;

    EXPECT_TRUE(sum[0] == std::complex<double>(12.0, 1.0));
    EXPECT_TRUE(diff[1] == std::complex<double>(6.0, 2.0));
    EXPECT_TRUE(product[0] == std::complex<double>(20.0, 10.0));
    EXPECT_TRUE(quotient[0] == scalar / std::complex<double>(2.0, 1.0));
}

TEST(OpsArithmetic, scalar_array_division_by_zero_throws)
{
    Vector<int> denominator{1, 0, 2};

    bool threw = false;

    try {
        10 / denominator;
    }
    catch (const Exceptions::StrataxError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsArithmetic, complex_scalar_array_division_by_zero_throws)
{
    Vector<std::complex<double>> denominator{
        {1.0, 0.0},
        {}
    };

    bool threw = false;

    try {
        std::complex<double>{10.0, 0.0} / denominator;
    }
    catch (const Exceptions::ZeroDivisionError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsArithmetic, compound_array_arithmetic)
{
    Vector<int> a{8, 12, 20};
    Vector<int> b{2, 3, 5};

    a += b;
    EXPECT_TRUE(a(0) == 10);
    EXPECT_TRUE(a(1) == 15);
    EXPECT_TRUE(a(2) == 25);

    a -= b;
    EXPECT_TRUE(a(0) == 8);
    EXPECT_TRUE(a(1) == 12);
    EXPECT_TRUE(a(2) == 20);

    a *= b;
    EXPECT_TRUE(a(0) == 16);
    EXPECT_TRUE(a(1) == 36);
    EXPECT_TRUE(a(2) == 100);

    a /= b;
    EXPECT_TRUE(a(0) == 8);
    EXPECT_TRUE(a(1) == 12);
    EXPECT_TRUE(a(2) == 20);
}

TEST(OpsArithmetic, compound_scalar_arithmetic)
{
    Vector<int> a{2, 4, 8};

    a += 2;
    EXPECT_TRUE(a(0) == 4);
    EXPECT_TRUE(a(2) == 10);

    a -= 2;
    EXPECT_TRUE(a(0) == 2);
    EXPECT_TRUE(a(2) == 8);

    a *= 2;
    EXPECT_TRUE(a(0) == 4);
    EXPECT_TRUE(a(2) == 16);

    a /= 2;
    EXPECT_TRUE(a(0) == 2);
    EXPECT_TRUE(a(2) == 8);
}

TEST(OpsArithmetic, complex_compound_scalar_arithmetic)
{
    Vector<std::complex<double>> a{
        {2.0, 0.0},
        {4.0, 0.0}
    };

    a += std::complex<double>{1.0, 1.0};
    EXPECT_TRUE(a[0] == std::complex<double>(3.0, 1.0));
    EXPECT_TRUE(a[1] == std::complex<double>(5.0, 1.0));

    a -= std::complex<double>{1.0, 1.0};
    EXPECT_TRUE(a[0] == std::complex<double>(2.0, 0.0));
    EXPECT_TRUE(a[1] == std::complex<double>(4.0, 0.0));

    a *= std::complex<double>{2.0, 0.0};
    EXPECT_TRUE(a[0] == std::complex<double>(4.0, 0.0));
    EXPECT_TRUE(a[1] == std::complex<double>(8.0, 0.0));

    a /= std::complex<double>{2.0, 0.0};
    EXPECT_TRUE(a[0] == std::complex<double>(2.0, 0.0));
    EXPECT_TRUE(a[1] == std::complex<double>(4.0, 0.0));
}

TEST(OpsArithmetic, unary_arithmetic)
{
    Vector<int> a{1, -2, 3};

    Vector<int> positive = +a;
    Vector<int> negative = -a;

    EXPECT_TRUE(positive(0) == 1);
    EXPECT_TRUE(positive(1) == -2);
    EXPECT_TRUE(positive(2) == 3);

    EXPECT_TRUE(negative(0) == -1);
    EXPECT_TRUE(negative(1) == 2);
    EXPECT_TRUE(negative(2) == -3);
}

TEST(OpsArithmetic, empty_array_arithmetic_preserves_shape)
{
    Vector<int> empty_vector(stratax::core::Shape{0});

    auto vector_sum = empty_vector + empty_vector;
    auto vector_scalar = empty_vector + 3;
    auto vector_unary = -empty_vector;

    EXPECT_TRUE(vector_sum.rank() == 1);
    EXPECT_TRUE(vector_sum.shape()(0) == 0);
    EXPECT_TRUE(vector_sum.empty());
    EXPECT_TRUE(vector_scalar.empty());
    EXPECT_TRUE(vector_unary.empty());

    Matrix<int> empty_matrix(0, 3);

    auto matrix_sum = empty_matrix + empty_matrix;
    auto matrix_scalar = empty_matrix * 2;

    EXPECT_TRUE(matrix_sum.rank() == 2);
    EXPECT_TRUE(matrix_sum.rows() == 0);
    EXPECT_TRUE(matrix_sum.cols() == 3);
    EXPECT_TRUE(matrix_sum.empty());
    EXPECT_TRUE(matrix_scalar.empty());
}

