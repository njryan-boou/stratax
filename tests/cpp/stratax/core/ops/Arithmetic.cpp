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

// ============================================================================
// Numeric Type Coverage (Float, Double)
// ============================================================================

TEST(OpsArithmetic, float_vector_arithmetic)
{
    Vector<float> a{2.5f, 4.5f, 8.5f};
    Vector<float> b{1.5f, 1.5f, 1.5f};

    Vector<float> sum = a + b;
    Vector<float> diff = a - b;
    Vector<float> product = a * b;
    Vector<float> quotient = a / b;

    EXPECT_TRUE(sum(0) == 4.0f);
    EXPECT_TRUE(diff(0) == 1.0f);
    EXPECT_TRUE(product(1) > 6.0f);
    EXPECT_TRUE(quotient(2) > 5.0f);
}

TEST(OpsArithmetic, double_matrix_arithmetic)
{
    Matrix<double> a(2, 2, 3.5);
    Matrix<double> b(2, 2, 2.5);

    Matrix<double> sum = a + b;
    Matrix<double> product = a * b;

    EXPECT_TRUE(sum(0, 0) > 5.9);
    EXPECT_TRUE(product(1, 1) > 8.7);
}

TEST(OpsArithmetic, float_scalar_arithmetic)
{
    Vector<float> a{1.5f, 2.5f, 3.5f};

    Vector<float> sum = a + 1.5f;
    Vector<float> product = a * 2.0f;

    EXPECT_TRUE(sum(0) > 2.9f);
    EXPECT_TRUE(product(2) > 6.9f);
}

TEST(OpsArithmetic, double_scalar_division)
{
    Vector<double> a{2.0, 4.0, 8.0};

    Vector<double> quotient = a / 2.0;

    EXPECT_TRUE(quotient(0) == 1.0);
    EXPECT_TRUE(quotient(1) == 2.0);
    EXPECT_TRUE(quotient(2) == 4.0);
}

// ============================================================================
// Edge Cases: Single Element Arrays
// ============================================================================

TEST(OpsArithmetic, single_element_vector_arithmetic)
{
    Vector<int> a(stratax::core::Shape{1});
    Vector<int> b(stratax::core::Shape{1});

    a(0) = 100;
    b(0) = 50;

    Vector<int> sum = a + b;
    Vector<int> diff = a - b;
    Vector<int> product = a * b;
    Vector<int> quotient = a / b;

    EXPECT_TRUE(sum(0) == 150);
    EXPECT_TRUE(diff(0) == 50);
    EXPECT_TRUE(product(0) == 5000);
    EXPECT_TRUE(quotient(0) == 2);
}

TEST(OpsArithmetic, single_element_matrix_arithmetic)
{
    Matrix<int> a(1, 1, 20);
    Matrix<int> b(1, 1, 5);

    Matrix<int> sum = a + b;
    Matrix<int> diff = a - b;
    Matrix<int> product = a * b;
    Matrix<int> quotient = a / b;

    EXPECT_TRUE(sum(0, 0) == 25);
    EXPECT_TRUE(diff(0, 0) == 15);
    EXPECT_TRUE(product(0, 0) == 100);
    EXPECT_TRUE(quotient(0, 0) == 4);
}

TEST(OpsArithmetic, single_element_tensor_arithmetic)
{
    Tensor<int> a(stratax::core::Shape{1, 1, 1}, 30);
    Tensor<int> b(stratax::core::Shape{1, 1, 1}, 10);

    Tensor<int> sum = a + b;
    Tensor<int> product = a * b;

    EXPECT_TRUE(sum(0) == 40);
    EXPECT_TRUE(product(0) == 300);
}

// ============================================================================
// Large Arrays
// ============================================================================

TEST(OpsArithmetic, large_vector_arithmetic)
{
    Vector<int> a(stratax::core::Shape{100});
    Vector<int> b(stratax::core::Shape{100});

    for (std::size_t i = 0; i < 100; ++i) {
        a(i) = static_cast<int>(i);
        b(i) = static_cast<int>(i + 1);
    }

    Vector<int> sum = a + b;

    EXPECT_TRUE(sum(0) == 1);
    EXPECT_TRUE(sum(99) == 199);
    EXPECT_TRUE(sum.size() == 100);
}

TEST(OpsArithmetic, large_matrix_arithmetic)
{
    Matrix<int> a(50, 50);
    Matrix<int> b(50, 50);

    for (std::size_t i = 0; i < 50 * 50; ++i) {
        a[i] = static_cast<int>(i);
        b[i] = static_cast<int>(i + 1);
    }

    Matrix<int> sum = a + b;

    EXPECT_TRUE(sum.rows() == 50);
    EXPECT_TRUE(sum.cols() == 50);
    EXPECT_TRUE(sum.size() == 2500);
}

// ============================================================================
// Commutative Properties
// ============================================================================

TEST(OpsArithmetic, addition_commutativity)
{
    Vector<int> a{1, 2, 3};
    Vector<int> b{4, 5, 6};

    Vector<int> sum1 = a + b;
    Vector<int> sum2 = b + a;

    EXPECT_TRUE(sum1(0) == sum2(0));
    EXPECT_TRUE(sum1(1) == sum2(1));
    EXPECT_TRUE(sum1(2) == sum2(2));
}

TEST(OpsArithmetic, multiplication_commutativity)
{
    Vector<int> a{2, 3, 4};
    Vector<int> b{5, 6, 7};

    Vector<int> product1 = a * b;
    Vector<int> product2 = b * a;

    EXPECT_TRUE(product1(0) == product2(0));
    EXPECT_TRUE(product1(1) == product2(1));
    EXPECT_TRUE(product1(2) == product2(2));
}

TEST(OpsArithmetic, scalar_multiplication_commutativity)
{
    Vector<int> a{2, 4, 6};

    Vector<int> left = 5 * a;
    Vector<int> right = a * 5;

    EXPECT_TRUE(left(0) == right(0));
    EXPECT_TRUE(left(1) == right(1));
    EXPECT_TRUE(left(2) == right(2));
}

// ============================================================================
// Zero and Negative Handling
// ============================================================================

TEST(OpsArithmetic, arithmetic_with_zeros)
{
    Vector<int> a{0, 1, 2};
    Vector<int> b{3, 0, 5};

    Vector<int> sum = a + b;
    Vector<int> diff = a - b;
    Vector<int> product = a * b;

    EXPECT_TRUE(sum(0) == 3);
    EXPECT_TRUE(diff(1) == 1);
    EXPECT_TRUE(product(2) == 10);
}

TEST(OpsArithmetic, all_negative_arithmetic)
{
    Vector<int> a{-1, -2, -3};
    Vector<int> b{-4, -5, -6};

    Vector<int> sum = a + b;
    Vector<int> product = a * b;

    EXPECT_TRUE(sum(0) == -5);
    EXPECT_TRUE(product(1) == 10);
    EXPECT_TRUE(product(2) == 18);
}

TEST(OpsArithmetic, mixed_sign_subtraction)
{
    Vector<int> a{5, -3, 0};
    Vector<int> b{-2, 4, 6};

    Vector<int> diff = a - b;

    EXPECT_TRUE(diff(0) == 7);
    EXPECT_TRUE(diff(1) == -7);
    EXPECT_TRUE(diff(2) == -6);
}

// ============================================================================
// Matrix Specific Operations
// ============================================================================

TEST(OpsArithmetic, rectangular_matrix_arithmetic)
{
    Matrix<int> a(3, 2);
    Matrix<int> b(3, 2);

    for (std::size_t i = 0; i < 6; ++i) {
        a[i] = static_cast<int>(i + 1);
        b[i] = static_cast<int>(i + 10);
    }

    Matrix<int> sum = a + b;

    EXPECT_TRUE(sum.rows() == 3);
    EXPECT_TRUE(sum.cols() == 2);
    EXPECT_TRUE(sum(0, 0) == 11);  // a[0]=1 + b[0]=10
    EXPECT_TRUE(sum(2, 1) == 21);  // a[5]=6 + b[5]=15
}

TEST(OpsArithmetic, single_row_matrix_arithmetic)
{
    Matrix<int> a(1, 5);
    Matrix<int> b(1, 5);

    for (std::size_t i = 0; i < 5; ++i) {
        a(0, i) = static_cast<int>(i);
        b(0, i) = static_cast<int>(i + 1);
    }

    Matrix<int> sum = a + b;

    EXPECT_TRUE(sum.rows() == 1);
    EXPECT_TRUE(sum.cols() == 5);
    EXPECT_TRUE(sum(0, 0) == 1);
    EXPECT_TRUE(sum(0, 4) == 9);
}

TEST(OpsArithmetic, single_column_matrix_arithmetic)
{
    Matrix<int> a(5, 1);
    Matrix<int> b(5, 1);

    for (std::size_t i = 0; i < 5; ++i) {
        a(i, 0) = static_cast<int>(i * 2);
        b(i, 0) = static_cast<int>(i * 3);
    }

    Matrix<int> sum = a + b;

    EXPECT_TRUE(sum.rows() == 5);
    EXPECT_TRUE(sum.cols() == 1);
    EXPECT_TRUE(sum(0, 0) == 0);
    EXPECT_TRUE(sum(4, 0) == 20);
}

// ============================================================================
// Tensor Operations
// ============================================================================

TEST(OpsArithmetic, tensor_rank_three_arithmetic)
{
    Tensor<int> a(stratax::core::Shape{2, 3, 4});
    Tensor<int> b(stratax::core::Shape{2, 3, 4});

    for (std::size_t i = 0; i < a.size(); ++i) {
        a(i) = static_cast<int>(i);
        b(i) = static_cast<int>(i + 1);
    }

    Tensor<int> sum = a + b;

    EXPECT_TRUE(sum.rank() == 3);
    EXPECT_TRUE(sum.size() == 24);
    EXPECT_TRUE(sum(0) == 1);
    EXPECT_TRUE(sum(23) == 47);
}

TEST(OpsArithmetic, tensor_high_rank_arithmetic)
{
    Tensor<int> a(stratax::core::Shape{2, 2, 2, 2});
    Tensor<int> b(stratax::core::Shape{2, 2, 2, 2});

    for (std::size_t i = 0; i < a.size(); ++i) {
        a(i) = static_cast<int>(i);
        b(i) = 1;
    }

    Tensor<int> sum = a + b;

    EXPECT_TRUE(sum.rank() == 4);
    EXPECT_TRUE(sum.size() == 16);
}

// ============================================================================
// Compound Assignment Edge Cases
// ============================================================================

TEST(OpsArithmetic, compound_addition_self)
{
    Vector<int> a{1, 2, 3};

    a += a;

    EXPECT_TRUE(a(0) == 2);
    EXPECT_TRUE(a(1) == 4);
    EXPECT_TRUE(a(2) == 6);
}

TEST(OpsArithmetic, compound_multiplication_self)
{
    Vector<int> a{2, 3, 4};

    a *= a;

    EXPECT_TRUE(a(0) == 4);
    EXPECT_TRUE(a(1) == 9);
    EXPECT_TRUE(a(2) == 16);
}

TEST(OpsArithmetic, compound_scalar_repeated)
{
    Vector<int> a{1, 2, 3};

    a += 1;
    a += 2;
    a *= 2;

    EXPECT_TRUE(a(0) == 8);
    EXPECT_TRUE(a(1) == 10);
    EXPECT_TRUE(a(2) == 12);
}

// ============================================================================
// Matrix Scalar Operations
// ============================================================================

TEST(OpsArithmetic, matrix_scalar_addition)
{
    Matrix<int> a(2, 2);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;

    Matrix<int> result = a + 10;

    EXPECT_TRUE(result(0, 0) == 11);
    EXPECT_TRUE(result(1, 1) == 14);
}

TEST(OpsArithmetic, matrix_scalar_multiplication)
{
    Matrix<int> a(2, 2);

    a(0, 0) = 2;
    a(0, 1) = 3;
    a(1, 0) = 4;
    a(1, 1) = 5;

    Matrix<int> result = a * 3;

    EXPECT_TRUE(result(0, 0) == 6);
    EXPECT_TRUE(result(1, 1) == 15);
}

TEST(OpsArithmetic, scalar_matrix_subtraction)
{
    Matrix<int> a(2, 2);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;

    Matrix<int> result = 10 - a;

    EXPECT_TRUE(result(0, 0) == 9);
    EXPECT_TRUE(result(1, 1) == 6);
}

// ============================================================================
// Tensor Scalar Operations
// ============================================================================

TEST(OpsArithmetic, tensor_scalar_arithmetic)
{
    Tensor<int> a(stratax::core::Shape{2, 2}, 5);

    Tensor<int> sum = a + 5;
    Tensor<int> product = a * 2;
    Tensor<int> diff = 20 - a;

    EXPECT_TRUE(sum(0) == 10);
    EXPECT_TRUE(product(1) == 10);
    EXPECT_TRUE(diff(2) == 15);
}

// ============================================================================
// Unary Operations Extended
// ============================================================================

TEST(OpsArithmetic, unary_on_matrix)
{
    Matrix<int> a(2, 2);

    a(0, 0) = 1;
    a(0, 1) = -2;
    a(1, 0) = 3;
    a(1, 1) = -4;

    Matrix<int> pos = +a;
    Matrix<int> neg = -a;

    EXPECT_TRUE(pos(0, 0) == 1);
    EXPECT_TRUE(neg(0, 1) == 2);
    EXPECT_TRUE(neg(1, 0) == -3);
}

TEST(OpsArithmetic, unary_on_tensor)
{
    Tensor<int> a(stratax::core::Shape{2, 2});

    a(0) = 1;
    a(1) = -2;
    a(2) = 3;
    a(3) = -4;

    Tensor<int> neg = -a;

    EXPECT_TRUE(neg(0) == -1);
    EXPECT_TRUE(neg(1) == 2);
    EXPECT_TRUE(neg(3) == 4);
}

// ============================================================================
// Floating Point Precision
// ============================================================================

TEST(OpsArithmetic, double_precision_addition)
{
    Vector<double> a{0.1, 0.2, 0.3};
    Vector<double> b{0.1, 0.2, 0.3};

    Vector<double> sum = a + b;

    EXPECT_TRUE(sum(0) > 0.19);
    EXPECT_TRUE(sum(0) < 0.21);
}

TEST(OpsArithmetic, double_precision_multiplication)
{
    Vector<double> a{0.5, 1.5, 2.5};
    Vector<double> b{2.0, 2.0, 2.0};

    Vector<double> product = a * b;

    EXPECT_TRUE(product(0) == 1.0);
    EXPECT_TRUE(product(1) == 3.0);
    EXPECT_TRUE(product(2) == 5.0);
}

// ============================================================================
// Compound Assignment on Different Containers
// ============================================================================

TEST(OpsArithmetic, matrix_compound_addition)
{
    Matrix<int> a(2, 2);
    Matrix<int> b(2, 2);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;

    b(0, 0) = 5;
    b(0, 1) = 6;
    b(1, 0) = 7;
    b(1, 1) = 8;

    a += b;

    EXPECT_TRUE(a(0, 0) == 6);
    EXPECT_TRUE(a(1, 1) == 12);
}

TEST(OpsArithmetic, tensor_compound_multiplication)
{
    Tensor<int> a(stratax::core::Shape{2, 2}, 2);
    Tensor<int> b(stratax::core::Shape{2, 2}, 3);

    a *= b;

    EXPECT_TRUE(a(0) == 6);
    EXPECT_TRUE(a(3) == 6);
}

// ============================================================================
// Division Result Verification
// ============================================================================

TEST(OpsArithmetic, integer_division_truncation)
{
    Vector<int> a{10, 15, 20};
    Vector<int> b{3, 3, 3};

    Vector<int> quotient = a / b;

    EXPECT_TRUE(quotient(0) == 3);
    EXPECT_TRUE(quotient(1) == 5);
    EXPECT_TRUE(quotient(2) == 6);
}

TEST(OpsArithmetic, scalar_division_precision)
{
    Vector<double> a{1.0, 2.0, 3.0};

    Vector<double> quotient = a / 2.0;

    EXPECT_TRUE(quotient(0) == 0.5);
    EXPECT_TRUE(quotient(1) == 1.0);
    EXPECT_TRUE(quotient(2) == 1.5);
}

