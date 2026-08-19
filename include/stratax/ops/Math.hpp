#pragma once

#include <cmath>
#include <type_traits>
#include <cstddef>

#include <stratax/core/ArrayTraits.hpp>
#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/core/dtypes/Types.hpp>
#include <stratax/ops/Broadcasting.hpp>

namespace stratax::core::math_detail {

// implementation machinery

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

template<typename T>
struct MathResult
{
    using type = std::remove_cvref_t<T>;
};

template<Integral T>
struct MathResult<T>
{
    using type = dtype::float64;
};

template<typename T>
using math_result_t =
    typename MathResult<std::remove_cvref_t<T>>::type;

template<Array A, typename Op>
auto standard_math_op(const A& arr, Op op)
{
    using result_value_type =
        math_result_t<typename A::value_type>;

    return unary_op<A, result_value_type>(arr, op);
}

template<typename L, typename R>
struct PowResult
{
    using promoted_type =
        promote_t<
            std::remove_cvref_t<L>,
            std::remove_cvref_t<R>>;

    using type = std::conditional_t<
        Integral<promoted_type>,
        dtype::float64,
        promoted_type>;
};

template<typename L, typename R>
using pow_result_t =
    typename PowResult<L, R>::type;

template<typename T>
struct AbsResult
{
    using type = std::remove_cvref_t<T>;
};

template<typename T>
requires (
    DTypeTraits<std::remove_cvref_t<T>>::kind ==
    DTypeKind::Complex
)
struct AbsResult<T>
{
    using type =
        complex_component_t<std::remove_cvref_t<T>>;
};

template<typename T>
using abs_result_t =
    typename AbsResult<std::remove_cvref_t<T>>::type;

} // namespace stratax::core::math_detail

namespace stratax::core {

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

} // namespace stratax::core
