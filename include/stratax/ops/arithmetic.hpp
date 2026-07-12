#pragma once

#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Validation.hpp>

/**
 * @brief Verifies that two arrays have the same shape before arithmetic.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @throws Exceptions::ShapeError If the operands do not match in size or shape.
 */
template<Array A>
void require_same_arithmetic_shape(const A& lhs, const A& rhs)
{
    stratax::core::validation::require_same_shape(
        lhs,
        rhs,
        "Arithmetic operands must have the same shape.");
}

/**
 * @brief Adds two arrays element by element.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return Array containing the element-wise sum.
 *
 * @throws Exceptions::ShapeError If the operands do not match in shape.
 */
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

/**
 * @brief Subtracts two arrays element by element.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return Array containing the element-wise difference.
 *
 * @throws Exceptions::ShapeError If the operands do not match in shape.
 */
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

/**
 * @brief Multiplies two arrays element by element.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return Array containing the element-wise product.
 *
 * @throws Exceptions::ShapeError If the operands do not match in shape.
 */
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

/**
 * @brief Divides two arrays element by element.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return Array containing the element-wise quotient.
 *
 * @throws Exceptions::ShapeError If the operands do not match in shape.
 * @throws Exceptions::ZeroDivisionError If any divisor element is zero.
 */
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
            throw Exceptions::ZeroDivisionError("Array division divisor element cannot be zero.");
        }

        *it3 = *it1 / *it2;
    }

    return result;
}

/**
 * @brief Adds a scalar to every element of an array.
 *
 * @param lhs Array operand.
 * @param rhs Scalar operand.
 *
 * @return Array containing the element-wise sum.
 */
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

/**
 * @brief Subtracts a scalar from every element of an array.
 *
 * @param lhs Array operand.
 * @param rhs Scalar operand.
 *
 * @return Array containing the element-wise difference.
 */
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

/**
 * @brief Multiplies every element of an array by a scalar.
 *
 * @param lhs Array operand.
 * @param rhs Scalar operand.
 *
 * @return Array containing the element-wise product.
 */
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

/**
 * @brief Divides every element of an array by a scalar.
 *
 * @param lhs Array operand.
 * @param rhs Scalar operand.
 *
 * @return Array containing the element-wise quotient.
 *
 * @throws Exceptions::ZeroDivisionError If the scalar is zero.
 */
template<Array A, Numeric Scalar>
A operator/(const A& lhs, const Scalar& rhs)
{
    A result(lhs.shape());

    auto out = result.begin();

    if (rhs == Scalar{})
    {
        throw Exceptions::ZeroDivisionError("Array division scalar divisor cannot be zero.");
    }

    for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
    {
        *out = *it / rhs;
    }

    return result;
}

/**
 * @brief Adds an array to a scalar.
 *
 * @param lhs Scalar operand.
 * @param rhs Array operand.
 *
 * @return Array containing the element-wise sum.
 */
template<Numeric Scalar, Array A>
A operator+(const Scalar& lhs, const A& rhs)
{
    return rhs + lhs;
}

/**
 * @brief Subtracts each array element from a scalar.
 *
 * @param lhs Scalar operand.
 * @param rhs Array operand.
 *
 * @return Array containing the element-wise difference.
 */
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

/**
 * @brief Multiplies an array by a scalar.
 *
 * @param lhs Scalar operand.
 * @param rhs Array operand.
 *
 * @return Array containing the element-wise product.
 */
template<Numeric Scalar, Array A>
A operator*(const Scalar& lhs, const A& rhs)
{
    return rhs * lhs;
}

/**
 * @brief Divides a scalar by each array element.
 *
 * @param lhs Scalar operand.
 * @param rhs Array operand.
 *
 * @return Array containing the element-wise quotient.
 *
 * @throws Exceptions::ZeroDivisionError If any array element is zero.
 */
template<Numeric Scalar, Array A>
A operator/(const Scalar& lhs, const A& rhs)
{
    A result(rhs.shape());

    auto out = result.begin();

    for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
    {
        if (*it == typename A::value_type{})
        {
            throw Exceptions::ZeroDivisionError("Scalar division divisor element cannot be zero.");
        }
        *out = lhs / *it;
    }

    return result;
}

/**
 * @brief Adds an array to itself in place.
 *
 * @param lhs Left-hand operand to modify.
 * @param rhs Right-hand operand.
 *
 * @return Reference to the updated array.
 * @throws Exceptions::ShapeError If the operands do not match in shape.
 */
template<Array A>
A& operator+=(A& lhs, const A& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

/**
 * @brief Subtracts an array from itself in place.
 *
 * @param lhs Left-hand operand to modify.
 * @param rhs Right-hand operand.
 *
 * @return Reference to the updated array.
 * @throws Exceptions::ShapeError If the operands do not match in shape.
 */
template<Array A>
A& operator-=(A& lhs, const A& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

/**
 * @brief Multiplies an array by another array in place.
 *
 * @param lhs Left-hand operand to modify.
 * @param rhs Right-hand operand.
 *
 * @return Reference to the updated array.
 * @throws Exceptions::ShapeError If the operands do not match in shape.
 */
template<Array A>
A& operator*=(A& lhs, const A& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

/**
 * @brief Divides an array by another array in place.
 *
 * @param lhs Left-hand operand to modify.
 * @param rhs Right-hand operand.
 *
 * @return Reference to the updated array.
 * @throws Exceptions::ShapeError If the operands do not match in shape.
 * @throws Exceptions::ZeroDivisionError If any divisor element is zero.
 */
template<Array A>
A& operator/=(A& lhs, const A& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

/**
 * @brief Adds a scalar to an array in place.
 *
 * @param lhs Array to modify.
 * @param rhs Scalar operand.
 *
 * @return Reference to the updated array.
 */
template<Array A, Numeric Scalar>
A& operator+=(A& lhs, const Scalar& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

/**
 * @brief Subtracts a scalar from an array in place.
 *
 * @param lhs Array to modify.
 * @param rhs Scalar operand.
 *
 * @return Reference to the updated array.
 */
template<Array A, Numeric Scalar>
A& operator-=(A& lhs, const Scalar& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

/**
 * @brief Multiplies an array by a scalar in place.
 *
 * @param lhs Array to modify.
 * @param rhs Scalar operand.
 *
 * @return Reference to the updated array.
 */
template<Array A, Numeric Scalar>
A& operator*=(A& lhs, const Scalar& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

/**
 * @brief Divides an array by a scalar in place.
 *
 * @param lhs Array to modify.
 * @param rhs Scalar operand.
 *
 * @return Reference to the updated array.
 * @throws Exceptions::ZeroDivisionError If the scalar is zero.
 */
template<Array A, Numeric Scalar>
A& operator/=(A& lhs, const Scalar& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

/**
 * @brief Negates every element of an array.
 *
 * @param arr Array operand.
 *
 * @return Array containing the negated values.
 */
template<Array A>
A operator-(const A& arr)
{
    return arr * typename A::value_type{-1};
}

/**
 * @brief Returns an array unchanged.
 *
 * @param arr Array operand.
 *
 * @return Copy of the input array.
 */
template<Array A>
A operator+(const A& arr)
{
    return arr;
}
