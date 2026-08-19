#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <concepts>
#include <type_traits>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

namespace
{

template<typename Actual, typename Expected>
void expect_near(const Actual& actual, const Expected& expected)
{
    ASSERT_EQ(actual.size(), expected.size());

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], 1e-12);
    }
}

template<typename T>
void expect_complex_near(
    const std::complex<T>& actual,
    const std::complex<T>& expected)
{
    EXPECT_NEAR(actual.real(), expected.real(), 1e-12);
    EXPECT_NEAR(actual.imag(), expected.imag(), 1e-12);
}

} // namespace

TEST(UnaryMath, RootsExponentialsAndLogarithms)
{
    const Vector<double> values{1.0, 4.0, 8.0};

    expect_near(sqrt(values), std::vector<double>{1.0, 2.0, std::sqrt(8.0)});
    expect_near(cbrt(values), std::vector<double>{1.0, std::cbrt(4.0), 2.0});
    expect_near(exp(values), std::vector<double>{std::exp(1.0), std::exp(4.0), std::exp(8.0)});
    expect_near(exp2(values), std::vector<double>{2.0, 16.0, 256.0});
    expect_near(log(values), std::vector<double>{0.0, std::log(4.0), std::log(8.0)});
    expect_near(log2(values), std::vector<double>{0.0, 2.0, 3.0});
    expect_near(log10(values), std::vector<double>{0.0, std::log10(4.0), std::log10(8.0)});
}

TEST(UnaryMath, AccurateSmallExponentialsAndLogarithms)
{
    const Vector<double> values{-0.25, 0.0, 0.25};
    const Vector<double> positive_values{0.5, 1.0, 4.0};

    expect_near(expm1(values), std::vector<double>{std::expm1(-0.25), 0.0, std::expm1(0.25)});
    expect_near(log1p(values), std::vector<double>{std::log1p(-0.25), 0.0, std::log1p(0.25)});
    expect_near(logb(positive_values), std::vector<double>{-1.0, 0.0, 2.0});
}

TEST(SpecialMath, ErrorAndGammaFunctions)
{
    const Vector<double> error_values{-0.5, 0.0, 0.5};
    const Vector<double> gamma_values{1.0, 2.0, 3.5};

    expect_near(erf(error_values), std::vector<double>{std::erf(-0.5), 0.0, std::erf(0.5)});
    expect_near(erfc(error_values), std::vector<double>{std::erfc(-0.5), 1.0, std::erfc(0.5)});
    expect_near(tgamma(gamma_values), std::vector<double>{1.0, 1.0, std::tgamma(3.5)});
    expect_near(lgamma(gamma_values), std::vector<double>{0.0, 0.0, std::lgamma(3.5)});
}

TEST(SpecialMath, PromotesIntegralInput)
{
    const Vector<stratax::dtype::int32> values{0, 1, 2};
    const auto result = expm1(values);

    static_assert(std::same_as<
        typename decltype(result)::value_type,
        stratax::dtype::float64>);
    expect_near(result, std::vector<double>{0.0, std::expm1(1.0), std::expm1(2.0)});
}

TEST(TrigonometricMath, DirectFunctions)
{
    const Vector<double> values{-0.5, 0.0, 0.5};

    expect_near(sin(values), std::vector<double>{std::sin(-0.5), 0.0, std::sin(0.5)});
    expect_near(cos(values), std::vector<double>{std::cos(-0.5), 1.0, std::cos(0.5)});
    expect_near(tan(values), std::vector<double>{std::tan(-0.5), 0.0, std::tan(0.5)});
}

TEST(TrigonometricMath, InverseFunctions)
{
    const Vector<double> values{-0.5, 0.0, 0.5};

    expect_near(asin(values), std::vector<double>{std::asin(-0.5), 0.0, std::asin(0.5)});
    expect_near(acos(values), std::vector<double>{std::acos(-0.5), std::acos(0.0), std::acos(0.5)});
    expect_near(atan(values), std::vector<double>{std::atan(-0.5), 0.0, std::atan(0.5)});
}

TEST(AbsoluteMath, RealAndComplexResultTypes)
{
    const Vector<double> real_values{-2.5, 0.0, 3.5};
    const Vector<stratax::dtype::complex128> complex_values{
        {3.0, 4.0},
        {5.0, 12.0}};

    const auto real_result = abs(real_values);
    const auto complex_result = abs(complex_values);

    static_assert(std::same_as<
        typename decltype(complex_result)::value_type,
        stratax::dtype::float64>);
    expect_near(real_result, std::vector<double>{2.5, 0.0, 3.5});
    expect_near(complex_result, std::vector<double>{5.0, 13.0});
}

TEST(RoundingMath, AppliesEveryRoundingMode)
{
    const Vector<double> values{-1.6, -1.4, 1.4, 1.6};

    expect_near(floor(values), std::vector<double>{-2.0, -2.0, 1.0, 1.0});
    expect_near(ceil(values), std::vector<double>{-1.0, -1.0, 2.0, 2.0});
    expect_near(trunc(values), std::vector<double>{-1.0, -1.0, 1.0, 1.0});
    expect_near(round(values), std::vector<double>{-2.0, -1.0, 1.0, 2.0});
    expect_near(nearbyint(values), std::vector<double>{-2.0, -1.0, 1.0, 2.0});
    expect_near(rint(values), std::vector<double>{-2.0, -1.0, 1.0, 2.0});
}

TEST(BinaryMath, AppliesEveryBinaryFunction)
{
    const Vector<double> lhs{5.5, -5.5, 2.0};
    const Vector<double> rhs{2.0, 2.0, -3.0};

    expect_near(pow(lhs, rhs), std::vector<double>{std::pow(5.5, 2.0), std::pow(-5.5, 2.0), 0.125});
    expect_near(atan2(lhs, rhs), std::vector<double>{std::atan2(5.5, 2.0), std::atan2(-5.5, 2.0), std::atan2(2.0, -3.0)});
    expect_near(hypot(lhs, rhs), std::vector<double>{std::hypot(5.5, 2.0), std::hypot(-5.5, 2.0), std::hypot(2.0, -3.0)});
    expect_near(fmod(lhs, rhs), std::vector<double>{std::fmod(5.5, 2.0), std::fmod(-5.5, 2.0), std::fmod(2.0, -3.0)});
    expect_near(remainder(lhs, rhs), std::vector<double>{std::remainder(5.5, 2.0), std::remainder(-5.5, 2.0), std::remainder(2.0, -3.0)});
    expect_near(copysign(lhs, rhs), std::vector<double>{5.5, 5.5, -2.0});
    expect_near(fmax(lhs, rhs), std::vector<double>{5.5, 2.0, 2.0});
    expect_near(fmin(lhs, rhs), std::vector<double>{2.0, -5.5, -3.0});
    expect_near(fdim(lhs, rhs), std::vector<double>{3.5, 0.0, 5.0});
}

TEST(BinaryMath, FindsNextRepresentableValues)
{
    const Vector<double> from{0.0, 1.0, -1.0};
    const Vector<double> toward{1.0, 2.0, -2.0};
    const auto result = nextafter(from, toward);

    EXPECT_EQ(result[0], std::nextafter(0.0, 1.0));
    EXPECT_EQ(result[1], std::nextafter(1.0, 2.0));
    EXPECT_EQ(result[2], std::nextafter(-1.0, -2.0));
}

TEST(BinaryMath, BroadcastsAndPromotesDtypes)
{
    const Matrix<stratax::dtype::float32> lhs{{1.0F, 2.0F}, {3.0F, 4.0F}};
    const Vector<stratax::dtype::int32> rhs{2, 3};
    const auto result = pow(lhs, rhs);

    static_assert(std::same_as<
        typename decltype(result)::value_type,
        stratax::dtype::float64>);
    static_assert(std::same_as<
        std::remove_cv_t<decltype(result)>,
        Tensor<stratax::dtype::float64>>);
    EXPECT_EQ(result.shape(), Shape({2, 2}));
    expect_near(result, std::vector<double>{1.0, 8.0, 9.0, 64.0});
}

TEST(UnaryMath, SupportsComplexStandardFunctions)
{
    using complex_type = stratax::dtype::complex128;
    const Vector<complex_type> values{{0.5, 0.25}, {1.0, -0.5}};
    const auto sqrt_result = sqrt(values);
    const auto exp_result = exp(values);
    const auto sin_result = sin(values);

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        expect_complex_near(sqrt_result[i], std::sqrt(values[i]));
        expect_complex_near(exp_result[i], std::exp(values[i]));
        expect_complex_near(sin_result[i], std::sin(values[i]));
    }
}

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
