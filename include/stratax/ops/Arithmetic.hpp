#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/ops/Broadcasting.hpp>

#include <functional>

/** @brief Applies a broadcasted element-wise binary operation to two arrays. */
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

/** @brief Applies an element-wise binary operation between an array and a scalar. */
template<Array A, Numeric Scalar, typename Op>
A binary_scalar_op(const A& lhs, const Scalar& rhs, Op op, bool check_zero_divisor = false)
{
	if (check_zero_divisor && rhs == Scalar{})
	{
		throw Exceptions::ZeroDivisionError("Array division scalar divisor cannot be zero.");
	}

	return broadcasted_op(lhs, rhs, op);
}

/** @brief Applies an element-wise binary operation between a scalar and an array. */
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

/** @brief Adds two arrays element by element. */
template<Array A>
A operator+(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts two arrays element by element. */
template<Array A>
A operator-(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies two arrays element by element. */
template<Array A>
A operator*(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides two arrays element by element. */
template<Array A>
A operator/(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a scalar to every element of an array. */
template<Array A, Numeric Scalar>
A operator+(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts a scalar from every element of an array. */
template<Array A, Numeric Scalar>
A operator-(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies every element of an array by a scalar. */
template<Array A, Numeric Scalar>
A operator*(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides every element of an array by a scalar. */
template<Array A, Numeric Scalar>
A operator/(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds an array to a scalar. */
template<Numeric Scalar, Array A>
A operator+(const Scalar& lhs, const A& rhs)
{
	return rhs + lhs;
}

/** @brief Subtracts each array element from a scalar. */
template<Numeric Scalar, Array A>
A operator-(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies an array by a scalar. */
template<Numeric Scalar, Array A>
A operator*(const Scalar& lhs, const A& rhs)
{
	return rhs * lhs;
}

/** @brief Divides a scalar by each array element. */
template<Numeric Scalar, Array A>
A operator/(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds an array to itself in place. */
template<Array A>
A& operator+=(A& lhs, const A& rhs)
{
	lhs = lhs + rhs;
	return lhs;
}

/** @brief Subtracts an array from itself in place. */
template<Array A>
A& operator-=(A& lhs, const A& rhs)
{
	lhs = lhs - rhs;
	return lhs;
}

/** @brief Multiplies an array by another array in place. */
template<Array A>
A& operator*=(A& lhs, const A& rhs)
{
	lhs = lhs * rhs;
	return lhs;
}

/** @brief Divides an array by another array in place. */
template<Array A>
A& operator/=(A& lhs, const A& rhs)
{
	lhs = lhs / rhs;
	return lhs;
}

/** @brief Adds a scalar to an array in place. */
template<Array A, Numeric Scalar>
A& operator+=(A& lhs, const Scalar& rhs)
{
	lhs = lhs + rhs;
	return lhs;
}

/** @brief Subtracts a scalar from an array in place. */
template<Array A, Numeric Scalar>
A& operator-=(A& lhs, const Scalar& rhs)
{
	lhs = lhs - rhs;
	return lhs;
}

/** @brief Multiplies an array by a scalar in place. */
template<Array A, Numeric Scalar>
A& operator*=(A& lhs, const Scalar& rhs)
{
	lhs = lhs * rhs;
	return lhs;
}

/** @brief Divides an array by a scalar in place. */
template<Array A, Numeric Scalar>
A& operator/=(A& lhs, const Scalar& rhs)
{
	lhs = lhs / rhs;
	return lhs;
}

/** @brief Negates every element of an array. */
template<Array A>
A operator-(const A& arr)
{
	return arr * typename A::value_type{-1};
}

/** @brief Returns an array unchanged. */
template<Array A>
A operator+(const A& arr)
{
	return arr;
}
