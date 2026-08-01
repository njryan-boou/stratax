#include <gtest/gtest.h>

#include <complex>

#include <stratax.h>

using namespace stratax::container;

TEST(OpsArithmetic, vector_elementwise_operators)
{
    const Vector<int> lhs{8, 12, 20};
    const Vector<int> rhs{2, 3, 5};

    const auto sum = lhs + rhs;
    const auto difference = lhs - rhs;
    const auto product = lhs * rhs;
    const auto quotient = lhs / rhs;

    EXPECT_EQ(sum[0], 10);
    EXPECT_EQ(sum[2], 25);
    EXPECT_EQ(difference[1], 9);
    EXPECT_EQ(product[2], 100);
    EXPECT_EQ(quotient[0], 4);
}

TEST(OpsArithmetic, matrix_elementwise_operators)
{
    const Matrix<int> lhs{{8, 12}, {20, 30}};
    const Matrix<int> rhs{{2, 3}, {5, 6}};

    const auto sum = lhs + rhs;
    const auto difference = lhs - rhs;
    const auto product = lhs * rhs;
    const auto quotient = lhs / rhs;

    EXPECT_EQ(sum.shape(), stratax::Shape({2, 2}));
    EXPECT_EQ(sum(1, 1), 36);
    EXPECT_EQ(difference(1, 1), 24);
    EXPECT_EQ(product(0, 1), 36);
    EXPECT_EQ(quotient(1, 0), 4);
}

TEST(OpsArithmetic, tensor_elementwise_operators)
{
    Tensor<int> lhs(stratax::Shape{2, 2});
    Tensor<int> rhs(stratax::Shape{2, 2});

    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        lhs[i] = static_cast<int>((i + 1) * 10);
        rhs[i] = static_cast<int>(i + 1);
    }

    EXPECT_EQ((lhs + rhs)[0], 11);
    EXPECT_EQ((lhs - rhs)[1], 18);
    EXPECT_EQ((lhs * rhs)[2], 90);
    EXPECT_EQ((lhs / rhs)[3], 10);
}

TEST(OpsArithmetic, array_and_scalar_operators_preserve_order)
{
    const Vector<int> values{2, 4, 8};

    EXPECT_EQ((values + 2)[2], 10);
    EXPECT_EQ((values - 2)[0], 0);
    EXPECT_EQ((values * 2)[1], 8);
    EXPECT_EQ((values / 2)[2], 4);

    EXPECT_EQ((10 + values)[0], 12);
    EXPECT_EQ((10 - values)[2], 2);
    EXPECT_EQ((10 * values)[1], 40);
    EXPECT_EQ((16 / values)[2], 2);
}

TEST(OpsArithmetic, zero_rhs_elements_are_allowed_for_non_division_array_ops)
{
    Vector<int> lhs{2, 4, 8};
    Vector<int> rhs{0, 2, 0};

    Vector<int> sum = lhs + rhs;
    Vector<int> diff = lhs - rhs;
    Vector<int> product = lhs * rhs;

    EXPECT_TRUE(sum[0] == 2);
    EXPECT_TRUE(sum[1] == 6);
    EXPECT_TRUE(diff[0] == 2);
    EXPECT_TRUE(diff[1] == 2);
    EXPECT_TRUE(product[0] == 0);
    EXPECT_TRUE(product[1] == 8);
    EXPECT_TRUE(product[2] == 0);
}

TEST(OpsArithmetic, complex_values_support_array_and_scalar_operations)
{
    using Complex = std::complex<double>;

    const Vector<Complex> lhs{{2.0, 1.0}, {4.0, -2.0}};
    const Vector<Complex> rhs{{1.0, 1.0}, {2.0, 0.0}};

    EXPECT_EQ((lhs + rhs)[0], Complex(3.0, 2.0));
    EXPECT_EQ((lhs * Complex(2.0, 0.0))[1], Complex(8.0, -4.0));
    EXPECT_EQ((Complex(10.0, 0.0) - lhs)[0], Complex(8.0, -1.0));
}

TEST(OpsArithmetic, division_by_zero_throws)
{
    const Vector<int> numerator{1, 2, 3};
    const Vector<int> zero_element{1, 0, 3};

    EXPECT_THROW(numerator / zero_element, Exceptions::ZeroDivisionError);
    EXPECT_THROW(numerator / 0, Exceptions::ZeroDivisionError);
    EXPECT_THROW(1 / zero_element, Exceptions::ZeroDivisionError);

    using Complex = std::complex<double>;
    const Vector<Complex> complex_values{{1.0, 0.0}, {2.0, 0.0}};
    const Vector<Complex> complex_zero{{1.0, 0.0}, {}};
    EXPECT_THROW(complex_values / complex_zero, Exceptions::ZeroDivisionError);
}

TEST(OpsArithmetic, zero_elements_are_valid_for_nondivision_operations)
{
    const Vector<int> lhs{2, 4, 8};
    const Vector<int> rhs{0, 2, 0};

    EXPECT_EQ((lhs + rhs)[0], 2);
    EXPECT_EQ((lhs - rhs)[2], 8);
    EXPECT_EQ((lhs * rhs)[2], 0);
}

TEST(OpsArithmetic, compound_array_operators)
{
    Vector<int> values{8, 12, 20};
    const Vector<int> operands{2, 3, 5};

    values += operands;
    EXPECT_EQ(values[2], 25);
    values -= operands;
    EXPECT_EQ(values[2], 20);
    values *= operands;
    EXPECT_EQ(values[1], 36);
    values /= operands;
    EXPECT_EQ(values[1], 12);
}

TEST(OpsArithmetic, compound_scalar_operators)
{
    Matrix<int> values{{4, 8}, {12, 16}};

    values += 2;
    values -= 1;
    values *= 2;
    values /= 2;

    EXPECT_EQ(values.shape(), stratax::Shape({2, 2}));
    EXPECT_EQ(values[0], 5);
    EXPECT_EQ(values[3], 17);
}

TEST(OpsArithmetic, unary_operators_work_for_each_container)
{
    const Vector<int> vector{-1, 2};
    const Matrix<int> matrix{{-3, 4}};
    const Tensor<int> tensor(stratax::Shape{2}, -5);

    EXPECT_EQ((+vector)[0], -1);
    EXPECT_EQ((-vector)[1], -2);
    EXPECT_EQ((-matrix)[0], 3);
    EXPECT_EQ((-tensor)[1], 5);
}

TEST(OpsArithmetic, empty_arrays_preserve_shape)
{
    const Vector<int> vector1;
    const Vector<int> vector2;
    const Matrix<int> matrix1;
    const Matrix<int> matrix2;

    EXPECT_TRUE((vector1 + vector2).empty());
    EXPECT_EQ((matrix1 * matrix2).shape(), stratax::Shape({0, 0}, stratax::Shape::allow_zero));
}

TEST(OpsArithmetic, floating_point_operations_preserve_precision)
{
    const Vector<double> lhs{0.1, 0.2, 1.0};
    const Vector<double> rhs{0.2, 0.3, 3.0};

    EXPECT_NEAR((lhs + rhs)[0], 0.3, 1e-12);
    EXPECT_NEAR((lhs * rhs)[1], 0.06, 1e-12);
    EXPECT_NEAR((lhs / rhs)[2], 1.0 / 3.0, 1e-12);
}

TEST(OpsArithmetic, high_rank_tensor_preserves_shape_and_values)
{
    const stratax::Shape shape{2, 2, 2, 2};
    const Tensor<int> lhs(shape, 3);
    const Tensor<int> rhs(shape, 4);

    const auto result = lhs * rhs;

    EXPECT_EQ(result.shape(), shape);
    EXPECT_EQ(result.size(), 16);
    EXPECT_EQ(result[0], 12);
    EXPECT_EQ(result[15], 12);
}

TEST(OpsArithmetic, integer_division_truncates)
{
    const Vector<int> lhs{5, -5, 7};
    const Vector<int> rhs{2, 2, 3};
    const auto result = lhs / rhs;

    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], -2);
    EXPECT_EQ(result[2], 2);
}
