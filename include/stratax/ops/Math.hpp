#pragma once

#include <cmath>
#include <type_traits>
#include <cstddef>

#include <stratax/core/ArrayTraits.hpp>
#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/core/dtypes/Types.hpp>
#include <stratax/ops/Broadcasting.hpp>

namespace stratax::core::math_detail {

/**
 * @brief Applies a unary callable to every element of an array.
 * @return Owning array with the input container and requested result dtype.
 * @invariant The result preserves the input shape and element order.
 * @internal
 */
template<Array A, DType Result, typename Op>
auto unary_op(const A& arr, Op op)
{
    using result_type =
        stratax::core::rebind_array_t<A, Result>;

    result_type result(arr.shape());

    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        result[i] = static_cast<Result>(op(arr[i]));
    }

    return result;
}

/** @brief Maps a non-integral input dtype to the same result dtype. @internal */
template<typename T>
struct MathResult
{
	/** @brief Result dtype for a standard unary math operation. */
    using type = std::remove_cvref_t<T>;
};

/** @brief Maps every supported integral dtype to float64. @internal */
template<Integral T>
struct MathResult<T>
{
	/** @brief Floating-point result dtype used for integral input. */
    using type = dtype::float64;
};

template<typename T>
using math_result_t =
    typename MathResult<std::remove_cvref_t<T>>::type;

/**
 * @brief Applies a unary math callable using the standard result dtype.
 * @return Owning array with the input container, shape, and inferred dtype.
 * @invariant Integral input is promoted to float64; other dtypes are preserved.
 * @internal
 */
template<Array A, typename Op>
auto standard_math_op(const A& arr, Op op)
{
    using result_value_type =
        math_result_t<typename A::value_type>;

    return unary_op<A, result_value_type>(arr, op);
}

/** @brief Determines the common result dtype for a binary math operation. @internal */
template<typename L, typename R>
struct PowResult
{
	/** @brief Dtype obtained from the ordinary Stratax promotion rules. */
    using promoted_type =
        promote_t<
            std::remove_cvref_t<L>,
            std::remove_cvref_t<R>>;

	/** @brief Promoted dtype, with integral-only results changed to float64. */
    using type = std::conditional_t<
        Integral<promoted_type>,
        dtype::float64,
        promoted_type>;
};

template<typename L, typename R>
using pow_result_t =
    typename PowResult<L, R>::type;

/** @brief Preserves the dtype of a real absolute-value operation. @internal */
template<typename T>
struct AbsResult
{
	/** @brief Absolute-value result dtype. */
    using type = std::remove_cvref_t<T>;
};

/** @brief Maps a complex dtype to its real component dtype. @internal */
template<typename T>
requires (
    DTypeTraits<std::remove_cvref_t<T>>::kind ==
    DTypeKind::Complex
)
struct AbsResult<T>
{
	/** @brief Real magnitude dtype corresponding to the complex input. */
    using type =
        complex_component_t<std::remove_cvref_t<T>>;
};

template<typename T>
using abs_result_t =
    typename AbsResult<std::remove_cvref_t<T>>::type;

} // namespace stratax::core::math_detail

namespace stratax::core {

/** @brief Computes the element-wise principal square root. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto sqrt(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::sqrt(value);
        });
}

/** @brief Computes the element-wise real cube root. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto cbrt(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::cbrt(value);
        });
}

/** @brief Computes e raised to each element. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto exp(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::exp(value);
        });
}

/** @brief Computes 2 raised to each element. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto exp2(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::exp2(value);
        });
}

/** @brief Computes e raised to each element minus one. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto expm1(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::expm1(value);
        });
}

/** @brief Computes the element-wise natural logarithm. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto log(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::log(value);
        });
}

/** @brief Computes the element-wise base-2 logarithm. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto log2(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::log2(value);
        });
}

/** @brief Computes the element-wise base-10 logarithm. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto log10(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::log10(value);
        });
}

/** @brief Computes the natural logarithm of one plus each element. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto log1p(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::log1p(value);
        });
}

/** @brief Extracts the unbiased floating-point exponent of each element. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto logb(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::logb(value);
        });
}

/** @brief Computes the element-wise sine in radians. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto sin(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::sin(value);
        });
}

/** @brief Computes the element-wise cosine in radians. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto cos(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::cos(value);
        });
}

/** @brief Computes the element-wise tangent in radians. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto tan(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::tan(value);
        });
}

/** @brief Computes the element-wise inverse sine. @return Shape-preserving owning array in radians. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto asin(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::asin(value);
        });
}

/** @brief Computes the element-wise inverse cosine. @return Shape-preserving owning array in radians. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto acos(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::acos(value);
        });
}

/** @brief Computes the element-wise inverse tangent. @return Shape-preserving owning array in radians. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto atan(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::atan(value);
        });
}

/** @brief Computes the element-wise hyperbolic sine. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto sinh(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::sinh(value);
        });
}

/** @brief Computes the element-wise hyperbolic cosine. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto cosh(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::cosh(value);
        });
}

/** @brief Computes the element-wise hyperbolic tangent. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto tanh(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::tanh(value);
        });
}

/** @brief Computes the element-wise inverse hyperbolic sine. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto asinh(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::asinh(value);
        });
}

/** @brief Computes the element-wise inverse hyperbolic cosine. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto acosh(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::acosh(value);
        });
}

/** @brief Computes the element-wise inverse hyperbolic tangent. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires Numeric<typename A::value_type>
auto atanh(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::atanh(value);
        });
}

/** @brief Computes the element-wise error function. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto erf(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::erf(value);
        });
}

/** @brief Computes the element-wise complementary error function. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto erfc(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::erfc(value);
        });
}

/** @brief Computes the element-wise gamma function. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto tgamma(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::tgamma(value);
        });
}

/** @brief Computes the logarithm of the absolute gamma function element-wise. @return Shape-preserving owning array. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto lgamma(const A& arr)
{
    return math_detail::standard_math_op(
        arr, [](const auto& value)
        {
            return std::lgamma(value);
        });
}

/**
 * @brief Computes the absolute value or complex magnitude of each element.
 * @return Shape-preserving array; complex input returns its component dtype.
 * @complexity O(arr.size()).
 */
template<Array A>
requires Numeric<typename A::value_type>
auto abs(const A& arr)
{
    using result_value_type =
        math_detail::abs_result_t<typename A::value_type>;

    return math_detail::unary_op<A, result_value_type>(
        arr,
        [](const auto& value)
        {
            using std::abs;
            return abs(value);
        });
}

/** @brief Rounds each element downward. @return Shape-preserving array with the input dtype. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto floor(const A& arr)
{
    using result_value_type = typename A::value_type;

    return math_detail::unary_op<A, result_value_type>(
        arr,
        [](const auto& value)
        {
            return std::floor(value);
        });
}

/** @brief Rounds each element upward. @return Shape-preserving array with the input dtype. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto ceil(const A& arr)
{
    using result_value_type = typename A::value_type;

    return math_detail::unary_op<A, result_value_type>(
        arr,
        [](const auto& value)
        {
            return std::ceil(value);
        });
}

/** @brief Rounds each element toward zero. @return Shape-preserving array with the input dtype. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto trunc(const A& arr)
{
    using result_value_type = typename A::value_type;

    return math_detail::unary_op<A, result_value_type>(
        arr,
        [](const auto& value)
        {
            return std::trunc(value);
        });
}

/** @brief Rounds each element to the nearest integer value. @return Shape-preserving array with the input dtype. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto round(const A& arr)
{
    using result_value_type = typename A::value_type;

    return math_detail::unary_op<A, result_value_type>(
        arr,
        [](const auto& value)
        {
            return std::round(value);
        });
}

/** @brief Rounds each element using the current rounding mode without raising inexact. @return Shape-preserving array with the input dtype. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto nearbyint(const A& arr)
{
    using result_value_type = typename A::value_type;

    return math_detail::unary_op<A, result_value_type>(
        arr,
        [](const auto& value)
        {
            return std::nearbyint(value);
        });
}

/** @brief Rounds each element using the current floating-point rounding mode. @return Shape-preserving array with the input dtype. @complexity O(arr.size()). */
template<Array A>
requires RealNumeric<typename A::value_type>
auto rint(const A& arr)
{
    using result_value_type = typename A::value_type;

    return math_detail::unary_op<A, result_value_type>(
        arr,
        [](const auto& value)
        {
            return std::rint(value);
        });
}

/**
 * @brief Raises each broadcasted left element to its corresponding right power.
 * @return Promoted owning array with the common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes are incompatible.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R>
auto pow(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::pow(a, b);
        });
}

/**
 * @brief Computes the two-argument arctangent of each broadcasted element pair.
 * @return Promoted owning array with the common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes are incompatible.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R>
auto atan2(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::atan2(a, b);
        });
}

/** @brief Computes element-wise hypotenuses after broadcasting. @return Promoted array with the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
auto hypot(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::hypot(a, b);
        });
}

/** @brief Computes element-wise floating-point remainders after broadcasting. @return Promoted array with the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
auto fmod(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::fmod(a, b);
        });
}

/** @brief Computes element-wise IEEE remainders after broadcasting. @return Promoted array with the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
auto remainder(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::remainder(a, b);
        });
}

/** @brief Copies each broadcasted right sign onto the corresponding left magnitude. @return Promoted array with the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
auto copysign(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::copysign(a, b);
        });
}

/** @brief Selects the element-wise floating-point maximum after broadcasting. @return Promoted array with the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
auto fmax(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::fmax(a, b);
        });
}

/** @brief Selects the element-wise floating-point minimum after broadcasting. @return Promoted array with the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
auto fmin(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::fmin(a, b);
        });
}

/** @brief Computes the positive difference of each broadcasted element pair. @return Promoted array with the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
requires (
    RealNumeric<typename L::value_type> &&
    RealNumeric<typename R::value_type>
)
auto fdim(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::fdim(a, b);
        });
}

/** @brief Finds the next representable value from each left element toward its broadcasted right element. @return Promoted array with the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
requires (
    RealNumeric<typename L::value_type> &&
    RealNumeric<typename R::value_type>
)
auto nextafter(const L& lhs, const R& rhs)
{
    using result_type =
        math_detail::pow_result_t<
            typename L::value_type,
            typename R::value_type>;

    return broadcasted_op<result_type>(
        lhs,
        rhs,
        [](const auto& a, const auto& b)
        {
            return std::nextafter(a, b);
        });
}

} // namespace stratax::core
