#pragma once

#include <stratax/core/Exceptions.hpp>

template<typename Array>
Array operator+(const Array& lhs, const Array& rhs)
{
    Array result(lhs.shape());

    auto it1 = lhs.begin();
    auto it2 = rhs.begin();
    auto it3 = result.begin();

    for (; it1 != lhs.end(); ++it1, ++it2, ++it3)
    {
        *it3 = *it1 + *it2;
    }

    return result;
}

template<typename Array>
Array operator-(const Array& lhs, const Array& rhs)
{
    Array result(lhs.shape());

    auto it1 = lhs.begin();
    auto it2 = rhs.begin();
    auto it3 = result.begin();

    for (; it1 != lhs.end(); ++it1, ++it2, ++it3)
    {
        *it3 = *it1 - *it2;
    }

    return result;
}

template<typename Array>
Array operator*(const Array& lhs, const Array& rhs)
{
    Array result(lhs.shape());

    auto it1 = lhs.begin();
    auto it2 = rhs.begin();
    auto it3 = result.begin();

    for (; it1 != lhs.end(); ++it1, ++it2, ++it3)
    {
        *it3 = *it1 * *it2;
    }

    return result;
}

template<typename Array>
Array operator/(const Array& lhs, const Array& rhs)
{
    Array result(lhs.shape());

    auto it1 = lhs.begin();
    auto it2 = rhs.begin();
    auto it3 = result.begin();

    for (; it1 != lhs.end(); ++it1, ++it2, ++it3)
    {
        if (*it2 == 0)
        {
            throw Exceptions::ZeroDivisionError("Division by zero");
        }

        *it3 = *it1 / *it2;
    }

    return result;
}

template<NDarray Array, Arithmetic Scalar>
Array operator+(const Array& lhs, const Scalar& rhs)
{
    Array result(lhs.shape());

    auto out = result.begin();

    for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
    {
        *out = *it + rhs;
    }

    return result;
}

template<NDarray Array, Arithmetic Scalar>
Array operator-(const Array& lhs, const Scalar& rhs)
{
    Array result(lhs.shape());

    auto out = result.begin();

    for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
    {
        *out = *it - rhs;
    }

    return result;
}

template<NDarray Array, Arithmetic Scalar>
Array operator*(const Array& lhs, const Scalar& rhs)
{
    Array result(lhs.shape());

    auto out = result.begin();

    for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
    {
        *out = *it * rhs;
    }

    return result;
}

template<NDarray Array, Arithmetic Scalar>
Array operator/(const Array& lhs, const Scalar& rhs)
{
    Array result(lhs.shape());

    auto out = result.begin();

    if (rhs == 0)
        {
            throw Exceptions::ZeroDivisionError("Division by zero");
        }

    for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
    {
        *out = *it / rhs;
    }

    return result;
}

template<Arithmetic Scalar, NDarray Array>
Array operator+(const Scalar& lhs, const Array& rhs)
{
    return rhs + lhs;
}

template<Arithmetic Scalar, NDarray Array>
Array operator-(const Scalar& lhs, const Array& rhs)
{
    Array result(rhs.shape());

    auto out = result.begin();

    for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
    {
        *out = lhs - *it;
    }

    return result;
}

template<Arithmetic Scalar, NDarray Array>
Array operator*(const Scalar& lhs, const Array& rhs)
{
    return rhs * lhs;
}

template<Arithmetic Scalar, NDarray Array>
Array operator/(const Scalar& lhs, const Array& rhs)
{
    Array result(rhs.shape());

    auto out = result.begin();

    for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
    {
        if (*it == 0)
        {
            throw Exceptions::ZeroDivisionError("Division by zero");
        }
        *out = lhs / *it;
    }

    return result;
}

template<NDarray Array>
Array& operator+=(Array& lhs, const Array& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

template<NDarray Array>
Array& operator-=(Array& lhs, const Array& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

template<NDarray Array>
Array& operator*=(Array& lhs, const Array& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

template<NDarray Array>
Array& operator/=(Array& lhs, const Array& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

template<NDarray Array, Arithmetic Scalar>
Array& operator+=(Array& lhs, const Scalar& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

template<NDarray Array, Arithmetic Scalar>
Array& operator-=(Array& lhs, const Scalar& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

template<NDarray Array, Arithmetic Scalar>
Array& operator*=(Array& lhs, const Scalar& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

template<NDarray Array, Arithmetic Scalar>
Array& operator/=(Array& lhs, const Scalar& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

template<NDarray Array>
Array operator-(const Array& arr)
{
    return arr * -1;
}

template<NDarray Array>
Array operator+(const Array& arr)
{
    return arr;
}
