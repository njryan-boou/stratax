#pragma once

#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/ops/Broadcasting.hpp>

#include <functional>

/**
 * @brief Applies a broadcasted element-wise binary operation to two arrays.
 *
 * @param lhs Left-hand array operand.
 * @param rhs Right-hand array operand.
 * @param op Callable used to combine each pair of elements.
 * @param check_zero_divisor Whether to reject zero values in the right-hand array.
 *
 * @return Array containing the element-wise operation result.
 *
 * @throws Exceptions::BroadcastError If the operand shapes are not broadcast-compatible.
 * @throws Exceptions::ZeroDivisionError If zero divisor checking is enabled and a divisor is zero.
 */
template<Array A, typename Op>
A binary_op(const A& lhs, const A& rhs, Op op, bool check_zero_divisor = false)
{
    auto checked_op = [&](const auto& left, const auto& right)
    {
        if (check_zero_divisor && right == typename A::value_type{})
        {
            throw Exceptions::ZeroDivisionError("Array division divisor element cannot be zero.");
        }

        return op(left, right);
    };

    return broadcasted_op(lhs, rhs, checked_op);
}

/**
 * @brief Applies an element-wise binary operation between an array and a scalar.
 *
 * @param lhs Array operand.
 * @param rhs Scalar operand.
 * @param op Callable used to combine each array element with the scalar.
 * @param check_zero_divisor Whether to reject a zero scalar divisor.
 *
 * @return Array containing the element-wise operation result.
 *
 * @throws Exceptions::ZeroDivisionError If zero divisor checking is enabled and the scalar is zero.
 */
template<Array A, Numeric Scalar, typename Op>
A binary_scalar_op(const A& lhs, const Scalar& rhs, Op op, bool check_zero_divisor = false)
{
    if (check_zero_divisor && rhs == Scalar{})
    {
        throw Exceptions::ZeroDivisionError("Array division scalar divisor cannot be zero.");
    }

    return broadcasted_op(lhs, rhs, op);
}

/**
 * @brief Applies an element-wise binary operation between a scalar and an array.
 *
 * @param lhs Scalar operand.
 * @param rhs Array operand.
 * @param op Callable used to combine the scalar with each array element.
 * @param check_zero_divisor Whether to reject zero values in the array divisor.
 *
 * @return Array containing the element-wise operation result.
 *
 * @throws Exceptions::ZeroDivisionError If zero divisor checking is enabled and a divisor is zero.
 */
template<Numeric Scalar, Array A, typename Op>
A binary_scalar_op(const Scalar& lhs, const A& rhs, Op op, bool check_zero_divisor = false)
{
    auto checked_op = [&](const auto& left, const auto& right)
    {
        if (check_zero_divisor && right == typename A::value_type{})
        {
            throw Exceptions::ZeroDivisionError("Scalar division divisor element cannot be zero.");
        }

        return op(left, right);
    };

    return broadcasted_op(lhs, rhs, checked_op);
}

/**
 * @brief Adds two arrays element by element.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return Array containing the element-wise sum.
 *
 * @throws Exceptions::BroadcastError If the operand shapes are not broadcast-compatible.
 */
template<Array A>
A operator+(const A& lhs, const A& rhs)
{
    return binary_op(lhs, rhs, std::plus<>{});
}

/**
 * @brief Subtracts two arrays element by element.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return Array containing the element-wise difference.
 *
 * @throws Exceptions::BroadcastError If the operand shapes are not broadcast-compatible.
 */
template<Array A>
A operator-(const A& lhs, const A& rhs)
{
    return binary_op(lhs, rhs, std::minus<>{});
}

/**
 * @brief Multiplies two arrays element by element.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return Array containing the element-wise product.
 *
 * @throws Exceptions::BroadcastError If the operand shapes are not broadcast-compatible.
 */
template<Array A>
A operator*(const A& lhs, const A& rhs)
{
    return binary_op(lhs, rhs, std::multiplies<>{});
}

/**
 * @brief Divides two arrays element by element.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return Array containing the element-wise quotient.
 *
 * @throws Exceptions::BroadcastError If the operand shapes are not broadcast-compatible.
 * @throws Exceptions::ZeroDivisionError If any divisor element is zero.
 */
template<Array A>
A operator/(const A& lhs, const A& rhs)
{
    return binary_op(lhs, rhs, std::divides<>{}, true);
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
    return binary_scalar_op(lhs, rhs, std::plus<>{});
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
    return binary_scalar_op(lhs, rhs, std::minus<>{});
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
    return binary_scalar_op(lhs, rhs, std::multiplies<>{});
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
    return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
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
    return binary_scalar_op(lhs, rhs, std::minus<>{});
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
    return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
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
