// TODO: Support mixed array/container and value types once
// result-container and dtype promotion rules are defined.

// TODO: Define whether compound assignment may change lhs shape through broadcasting.
#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/ops/Broadcasting.hpp>

#include <functional>

/**
 * @brief Applies an arithmetic callable to two broadcast-compatible arrays.
 *
 * The result has the common broadcasted shape and owns independent storage.
 * When @p check_zero_divisor is true, each broadcasted right-hand value is
 * checked before invoking @p op.
 *
 * @tparam A Common Stratax array and result type.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Left array operand.
 * @param rhs Right array operand.
 * @param op Callable invoked for every broadcasted element pair.
 * @param check_zero_divisor Whether zero right-hand values are rejected.
 * @return Owning array containing the element-wise results.
 * @throws Exceptions::BroadcastError If the shapes are incompatible.
 * @throws Exceptions::ZeroDivisionError If divisor checking is enabled and a
 *         broadcasted right-hand element equals zero.
 * @throws Any exception propagated by allocation or @p op.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
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
 * @brief Applies an arithmetic callable to an array and a right scalar.
 * @tparam A Stratax array and result type.
 * @tparam Scalar Numeric scalar type.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Array supplying each left argument.
 * @param rhs Scalar supplied as every right argument.
 * @param op Callable invoked for each array element.
 * @param check_zero_divisor Whether a zero @p rhs is rejected.
 * @return Owning array with the same shape as @p lhs.
 * @throws Exceptions::ZeroDivisionError If divisor checking is enabled and
 *         @p rhs equals zero, including when @p lhs is empty.
 * @throws Any exception propagated by allocation or @p op.
 * @complexity O(lhs.size()).
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
 * @brief Applies an arithmetic callable to a left scalar and an array.
 * @tparam Scalar Numeric scalar type.
 * @tparam A Stratax array and result type.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Scalar supplied as every left argument.
 * @param rhs Array supplying each right argument.
 * @param op Callable invoked for each array element.
 * @param check_zero_divisor Whether zero elements of @p rhs are rejected.
 * @return Owning array with the same shape as @p rhs.
 * @throws Exceptions::ZeroDivisionError If divisor checking is enabled and an
 *         element of @p rhs equals zero.
 * @throws Any exception propagated by allocation or @p op.
 * @complexity O(rhs.size()).
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

/** @brief Adds two broadcast-compatible arrays. @return Owning broadcasted sum. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array A>
A operator+(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts two broadcast-compatible arrays. @return Owning `lhs - rhs` result. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array A>
A operator-(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies two broadcast-compatible arrays element-wise. @return Owning broadcasted product. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array A>
A operator*(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides two broadcast-compatible arrays element-wise. @return Owning broadcasted quotient. @throws Exceptions::BroadcastError If shapes are incompatible. @throws Exceptions::ZeroDivisionError If a used divisor element is zero. @complexity O(n * r). */
template<Array A>
A operator/(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a scalar to every array element. @return Owning result with the array shape. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
A operator+(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts a scalar from every array element. @return Owning result with the array shape. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
A operator-(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies every array element by a scalar. @return Owning result with the array shape. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
A operator*(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides every array element by a scalar. @return Owning result with the array shape. @throws Exceptions::ZeroDivisionError If @p rhs is zero. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
A operator/(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a scalar to every array element. @return Owning result with the array shape. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
A operator+(const Scalar& lhs, const A& rhs)
{
	return rhs + lhs;
}

/** @brief Subtracts every array element from a scalar. @return Owning `lhs - rhs[i]` result. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
A operator-(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies every array element by a scalar. @return Owning result with the array shape. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
A operator*(const Scalar& lhs, const A& rhs)
{
	return rhs * lhs;
}

/** @brief Divides a scalar by every array element. @return Owning `lhs / rhs[i]` result. @throws Exceptions::ZeroDivisionError If any divisor element is zero. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
A operator/(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a broadcast-compatible array and assigns the result to @p lhs. @return Reference to @p lhs, whose shape may become the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array A>
A& operator+=(A& lhs, const A& rhs)
{
	lhs = lhs + rhs;
	return lhs;
}

/** @brief Subtracts a broadcast-compatible array and assigns the result to @p lhs. @return Reference to @p lhs, whose shape may become the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array A>
A& operator-=(A& lhs, const A& rhs)
{
	lhs = lhs - rhs;
	return lhs;
}

/** @brief Multiplies by a broadcast-compatible array and assigns to @p lhs. @return Reference to @p lhs, whose shape may become the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array A>
A& operator*=(A& lhs, const A& rhs)
{
	lhs = lhs * rhs;
	return lhs;
}

/** @brief Divides by a broadcast-compatible array and assigns to @p lhs. @return Reference to @p lhs, whose shape may become the broadcasted shape. @throws Exceptions::BroadcastError If shapes are incompatible. @throws Exceptions::ZeroDivisionError If a used divisor is zero. @complexity O(n * r). */
template<Array A>
A& operator/=(A& lhs, const A& rhs)
{
	lhs = lhs / rhs;
	return lhs;
}

/** @brief Adds a scalar to @p lhs in place. @return Reference to @p lhs. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
A& operator+=(A& lhs, const Scalar& rhs)
{
	lhs = lhs + rhs;
	return lhs;
}

/** @brief Subtracts a scalar from @p lhs in place. @return Reference to @p lhs. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
A& operator-=(A& lhs, const Scalar& rhs)
{
	lhs = lhs - rhs;
	return lhs;
}

/** @brief Multiplies @p lhs by a scalar in place. @return Reference to @p lhs. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
A& operator*=(A& lhs, const Scalar& rhs)
{
	lhs = lhs * rhs;
	return lhs;
}

/** @brief Divides @p lhs by a scalar in place. @return Reference to @p lhs. @throws Exceptions::ZeroDivisionError If @p rhs is zero. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
A& operator/=(A& lhs, const Scalar& rhs)
{
	lhs = lhs / rhs;
	return lhs;
}

/** @brief Returns an owning element-wise negated copy of @p arr. @complexity O(arr.size()). */
template<Array A>
A operator-(const A& arr)
{
	return arr * typename A::value_type{-1};
}

/** @brief Returns an owning unchanged copy of @p arr. @complexity O(arr.size()). */
template<Array A>
A operator+(const A& arr)
{
	return arr;
}
