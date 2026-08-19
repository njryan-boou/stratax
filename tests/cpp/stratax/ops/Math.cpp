#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <type_traits>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(HyperbolicMath, AppliesFunctionsElementWise)
{
    const Vector<double> values{0.25, 0.5, 0.75};

    const auto sinh_result = sinh(values);
    const auto cosh_result = cosh(values);
    const auto tanh_result = tanh(values);
    const auto asinh_result = asinh(values);

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(sinh_result[i], std::sinh(values[i]));
        EXPECT_DOUBLE_EQ(cosh_result[i], std::cosh(values[i]));
        EXPECT_DOUBLE_EQ(tanh_result[i], std::tanh(values[i]));
        EXPECT_DOUBLE_EQ(asinh_result[i], std::asinh(values[i]));
    }
}

TEST(HyperbolicMath, AppliesInverseFunctionsWithinRealDomains)
{
    const Vector<double> acosh_values{1.0, 2.0, 4.0};
    const Vector<double> atanh_values{-0.5, 0.0, 0.5};
    const auto acosh_result = acosh(acosh_values);
    const auto atanh_result = atanh(atanh_values);

    for (std::size_t i = 0; i < acosh_values.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(acosh_result[i], std::acosh(acosh_values[i]));
        EXPECT_DOUBLE_EQ(atanh_result[i], std::atanh(atanh_values[i]));
    }
}

TEST(HyperbolicMath, PromotesIntegralElementsToFloat64)
{
    const Vector<stratax::dtype::int32> values{0, 1, 2};
    const auto result = sinh(values);

    static_assert(std::same_as<
        typename decltype(result)::value_type,
        stratax::dtype::float64>);
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], std::sinh(1.0));
    EXPECT_DOUBLE_EQ(result[2], std::sinh(2.0));
}

TEST(HyperbolicMath, PreservesContainerShape)
{
    const Matrix<double> values{{0.0, 0.5}, {1.0, 1.5}};
    const auto result = cosh(values);

    static_assert(std::same_as<
        std::remove_cv_t<decltype(result)>,
        Matrix<double>>);
    EXPECT_EQ(result.shape(), values.shape());
    EXPECT_DOUBLE_EQ(result[0], std::cosh(0.0));
    EXPECT_DOUBLE_EQ(result[3], std::cosh(1.5));
}

TEST(HyperbolicMath, SupportsComplexElements)
{
    using complex_type = stratax::dtype::complex128;
    const Vector<complex_type> values{
        complex_type{0.5, 0.25},
        complex_type{1.0, -0.5}};
    const auto result = tanh(values);

    static_assert(std::same_as<
        std::remove_cv_t<decltype(result)>,
        Vector<complex_type>>);
    EXPECT_EQ(result[0], std::tanh(values[0]));
    EXPECT_EQ(result[1], std::tanh(values[1]));
}

TEST(HyperbolicMath, PreservesEmptyShape)
{
    const Tensor<double> values(Shape{2, 0, 3});

    EXPECT_EQ(asinh(values).shape(), values.shape());
    EXPECT_TRUE(asinh(values).empty());
}
