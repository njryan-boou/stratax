// TODO: Support mixed array/container types once result promotion rules are defined.

// TODO: Define whether compound assignment may change lhs shape through broadcasting.
#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/ops/Broadcasting.hpp>

#include <functional>

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

template<Array A, Numeric Scalar, typename Op>
A binary_scalar_op(const A& lhs, const Scalar& rhs, Op op, bool check_zero_divisor = false)
{
	if (check_zero_divisor && rhs == Scalar{})
	{
		throw Exceptions::ZeroDivisionError("Array division scalar divisor cannot be zero.");
	}

	return broadcasted_op(lhs, rhs, op);
}

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

template<Array A>
A operator+(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::plus<>{});
}

template<Array A>
A operator-(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::minus<>{});
}

template<Array A>
A operator*(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::multiplies<>{});
}

template<Array A>
A operator/(const A& lhs, const A& rhs)
{
	return binary_op(lhs, rhs, std::divides<>{}, true);
}

template<Array A, Numeric Scalar>
A operator+(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::plus<>{});
}

template<Array A, Numeric Scalar>
A operator-(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::minus<>{});
}

template<Array A, Numeric Scalar>
A operator*(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::multiplies<>{});
}

template<Array A, Numeric Scalar>
A operator/(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
}

template<Numeric Scalar, Array A>
A operator+(const Scalar& lhs, const A& rhs)
{
	return rhs + lhs;
}

template<Numeric Scalar, Array A>
A operator-(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_op(lhs, rhs, std::minus<>{});
}

template<Numeric Scalar, Array A>
A operator*(const Scalar& lhs, const A& rhs)
{
	return rhs * lhs;
}

template<Numeric Scalar, Array A>
A operator/(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
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
