#pragma once

#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>

template<Array A>
void require_same_arithmetic_shape(const A& lhs, const A& rhs)
{
    if (lhs.size() != rhs.size() || lhs.shape() != rhs.shape())
    {
        throw Exceptions::ShapeError("Arithmetic operands must have the same shape.");
    }
}

template<Array A>
A operator+(const A& lhs, const A& rhs)
{
    require_same_arithmetic_shape(lhs, rhs);

    A result(lhs.shape());

    auto it1 = lhs.begin();
    auto it2 = rhs.begin();
    auto it3 = result.begin();

    for (; it1 != lhs.end(); ++it1, ++it2, ++it3)
    {
        *it3 = *it1 + *it2;
    }

    return result;
}

template<Array A>
A operator-(const A& lhs, const A& rhs)
{
    require_same_arithmetic_shape(lhs, rhs);

    A result(lhs.shape());

    auto it1 = lhs.begin();
    auto it2 = rhs.begin();
    auto it3 = result.begin();

    for (; it1 != lhs.end(); ++it1, ++it2, ++it3)
    {
        *it3 = *it1 - *it2;
    }

    return result;
}

template<Array A>
A operator*(const A& lhs, const A& rhs)
{
    require_same_arithmetic_shape(lhs, rhs);

    A result(lhs.shape());

    auto it1 = lhs.begin();
    auto it2 = rhs.begin();
    auto it3 = result.begin();

    for (; it1 != lhs.end(); ++it1, ++it2, ++it3)
    {
        *it3 = *it1 * *it2;
    }

    return result;
}

template<Array A>
A operator/(const A& lhs, const A& rhs)
{
    require_same_arithmetic_shape(lhs, rhs);

    A result(lhs.shape());

    auto it1 = lhs.begin();
    auto it2 = rhs.begin();
    auto it3 = result.begin();

    for (; it1 != lhs.end(); ++it1, ++it2, ++it3)
    {
        if (*it2 == typename A::value_type{})
        {
            throw Exceptions::ZeroDivisionError("Division by zero");
        }

        *it3 = *it1 / *it2;
    }

    return result;
}

template<Array A, Numeric Scalar>
A operator+(const A& lhs, const Scalar& rhs)
{
    A result(lhs.shape());

    auto out = result.begin();

    for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
    {
        *out = *it + rhs;
    }

    return result;
}

template<Array A, Numeric Scalar>
A operator-(const A& lhs, const Scalar& rhs)
{
    A result(lhs.shape());

    auto out = result.begin();

    for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
    {
        *out = *it - rhs;
    }

    return result;
}

template<Array A, Numeric Scalar>
A operator*(const A& lhs, const Scalar& rhs)
{
    A result(lhs.shape());

    auto out = result.begin();

    for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
    {
        *out = *it * rhs;
    }

    return result;
}

template<Array A, Numeric Scalar>
A operator/(const A& lhs, const Scalar& rhs)
{
    A result(lhs.shape());

    auto out = result.begin();

    if (rhs == Scalar{})
    {
        throw Exceptions::ZeroDivisionError("Division by zero");
    }

    for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
    {
        *out = *it / rhs;
    }

    return result;
}

template<Numeric Scalar, Array A>
A operator+(const Scalar& lhs, const A& rhs)
{
    return rhs + lhs;
}

template<Numeric Scalar, Array A>
A operator-(const Scalar& lhs, const A& rhs)
{
    A result(rhs.shape());

    auto out = result.begin();

    for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
    {
        *out = lhs - *it;
    }

    return result;
}

template<Numeric Scalar, Array A>
A operator*(const Scalar& lhs, const A& rhs)
{
    return rhs * lhs;
}

template<Numeric Scalar, Array A>
A operator/(const Scalar& lhs, const A& rhs)
{
    A result(rhs.shape());

    auto out = result.begin();

    for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
    {
        if (*it == typename A::value_type{})
        {
            throw Exceptions::ZeroDivisionError("Division by zero");
        }
        *out = lhs / *it;
    }

    return result;
}

template<Array A>
A& operator+=(A& lhs, const A& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

template<Array A>
A& operator-=(A& lhs, const A& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

template<Array A>
A& operator*=(A& lhs, const A& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

template<Array A>
A& operator/=(A& lhs, const A& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

template<Array A, Numeric Scalar>
A& operator+=(A& lhs, const Scalar& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

template<Array A, Numeric Scalar>
A& operator-=(A& lhs, const Scalar& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

template<Array A, Numeric Scalar>
A& operator*=(A& lhs, const Scalar& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

template<Array A, Numeric Scalar>
A& operator/=(A& lhs, const Scalar& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

template<Array A>
A operator-(const A& arr)
{
    return arr * typename A::value_type{-1};
}

template<Array A>
A operator+(const A& arr)
{
    return arr;
}
