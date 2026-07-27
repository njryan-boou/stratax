#pragma once

#include <stratax/core/Concepts.hpp>
#include <stratax/core/validation/Validation.hpp>

/**
 * @brief Verifies that two arrays have the same shape before bitwise operations.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 */
template<Array A>
requires Integer<typename A::value_type>
void require_same_bitwise_shape(const A& lhs, const A& rhs)
{
	stratax::core::validation::require_same_shape(
		lhs,
		rhs,
		"Bitwise operands must have the same shape.");
}

// Unary

/**
 * @brief Applies bitwise NOT to each element of an integer array.
 *
 * @param value Input array.
 * @return Array with element-wise bitwise NOT results.
 */
template<Array A>
requires Integer<typename A::value_type>
A operator~(const A& value)
{
	A result(value.shape());

	auto out = result.begin();
	for (auto it = value.begin(); it != value.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(~(*it));
	}

	return result;
}

// Array-array

/**
 * @brief Applies element-wise bitwise AND between two arrays.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 * @return Array with element-wise bitwise AND results.
 */
template<Array A>
requires Integer<typename A::value_type>
A operator&(const A& lhs, const A& rhs)
{
	require_same_bitwise_shape(lhs, rhs);

	A result(lhs.shape());

	auto out = result.begin();
	auto it_lhs = lhs.begin();
	auto it_rhs = rhs.begin();

	for (; it_lhs != lhs.end(); ++it_lhs, ++it_rhs, ++out)
	{
		*out = static_cast<typename A::value_type>(*it_lhs & *it_rhs);
	}

	return result;
}

/**
 * @brief Applies element-wise bitwise OR between two arrays.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 * @return Array with element-wise bitwise OR results.
 */
template<Array A>
requires Integer<typename A::value_type>
A operator|(const A& lhs, const A& rhs)
{
	require_same_bitwise_shape(lhs, rhs);

	A result(lhs.shape());

	auto out = result.begin();
	auto it_lhs = lhs.begin();
	auto it_rhs = rhs.begin();

	for (; it_lhs != lhs.end(); ++it_lhs, ++it_rhs, ++out)
	{
		*out = static_cast<typename A::value_type>(*it_lhs | *it_rhs);
	}

	return result;
}

/**
 * @brief Applies element-wise bitwise XOR between two arrays.
 *
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 * @return Array with element-wise bitwise XOR results.
 */
template<Array A>
requires Integer<typename A::value_type>
A operator^(const A& lhs, const A& rhs)
{
	require_same_bitwise_shape(lhs, rhs);

	A result(lhs.shape());

	auto out = result.begin();
	auto it_lhs = lhs.begin();
	auto it_rhs = rhs.begin();

	for (; it_lhs != lhs.end(); ++it_lhs, ++it_rhs, ++out)
	{
		*out = static_cast<typename A::value_type>(*it_lhs ^ *it_rhs);
	}

	return result;
}

/**
 * @brief Applies element-wise left shift using per-element shift counts.
 *
 * @param lhs Value operand array.
 * @param rhs Shift-count operand array.
 * @return Array with element-wise left-shift results.
 */
template<Array A>
requires Integer<typename A::value_type>
A operator<<(const A& lhs, const A& rhs)
{
	require_same_bitwise_shape(lhs, rhs);

	A result(lhs.shape());

	auto out = result.begin();
	auto it_lhs = lhs.begin();
	auto it_rhs = rhs.begin();

	for (; it_lhs != lhs.end(); ++it_lhs, ++it_rhs, ++out)
	{
		*out = static_cast<typename A::value_type>(*it_lhs << *it_rhs);
	}

	return result;
}

/**
 * @brief Applies element-wise right shift using per-element shift counts.
 *
 * @param lhs Value operand array.
 * @param rhs Shift-count operand array.
 * @return Array with element-wise right-shift results.
 */
template<Array A>
requires Integer<typename A::value_type>
A operator>>(const A& lhs, const A& rhs)
{
	require_same_bitwise_shape(lhs, rhs);

	A result(lhs.shape());

	auto out = result.begin();
	auto it_lhs = lhs.begin();
	auto it_rhs = rhs.begin();

	for (; it_lhs != lhs.end(); ++it_lhs, ++it_rhs, ++out)
	{
		*out = static_cast<typename A::value_type>(*it_lhs >> *it_rhs);
	}

	return result;
}

// Array-scalar

/**
 * @brief Applies bitwise AND between each array element and a scalar.
 *
 * @param lhs Array operand.
 * @param rhs Scalar operand.
 * @return Array with element-wise bitwise AND results.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator&(const A& lhs, const Scalar& rhs)
{
	A result(lhs.shape());

	auto out = result.begin();
	for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(*it & rhs);
	}

	return result;
}

/**
 * @brief Applies bitwise OR between each array element and a scalar.
 *
 * @param lhs Array operand.
 * @param rhs Scalar operand.
 * @return Array with element-wise bitwise OR results.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator|(const A& lhs, const Scalar& rhs)
{
	A result(lhs.shape());

	auto out = result.begin();
	for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(*it | rhs);
	}

	return result;
}

/**
 * @brief Applies bitwise XOR between each array element and a scalar.
 *
 * @param lhs Array operand.
 * @param rhs Scalar operand.
 * @return Array with element-wise bitwise XOR results.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator^(const A& lhs, const Scalar& rhs)
{
	A result(lhs.shape());

	auto out = result.begin();
	for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(*it ^ rhs);
	}

	return result;
}

/**
 * @brief Left-shifts each array element by a scalar shift count.
 *
 * @param lhs Array operand.
 * @param rhs Scalar shift count.
 * @return Array with element-wise left-shift results.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator<<(const A& lhs, const Scalar& rhs)
{
	A result(lhs.shape());

	auto out = result.begin();
	for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(*it << rhs);
	}

	return result;
}

/**
 * @brief Right-shifts each array element by a scalar shift count.
 *
 * @param lhs Array operand.
 * @param rhs Scalar shift count.
 * @return Array with element-wise right-shift results.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator>>(const A& lhs, const Scalar& rhs)
{
	A result(lhs.shape());

	auto out = result.begin();
	for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(*it >> rhs);
	}

	return result;
}

// Scalar-array (reverse)

/**
 * @brief Applies bitwise AND between a scalar and each array element.
 *
 * @param lhs Scalar operand.
 * @param rhs Array operand.
 * @return Array with element-wise bitwise AND results.
 */
template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator&(const Scalar& lhs, const A& rhs)
{
	return rhs & lhs;
}

/**
 * @brief Applies bitwise OR between a scalar and each array element.
 *
 * @param lhs Scalar operand.
 * @param rhs Array operand.
 * @return Array with element-wise bitwise OR results.
 */
template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator|(const Scalar& lhs, const A& rhs)
{
	return rhs | lhs;
}

/**
 * @brief Applies bitwise XOR between a scalar and each array element.
 *
 * @param lhs Scalar operand.
 * @param rhs Array operand.
 * @return Array with element-wise bitwise XOR results.
 */
template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator^(const Scalar& lhs, const A& rhs)
{
	return rhs ^ lhs;
}

/**
 * @brief Left-shifts a scalar value by each array element as shift count.
 *
 * @param lhs Scalar value operand.
 * @param rhs Array shift-count operand.
 * @return Array with element-wise left-shift results.
 */
template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator<<(const Scalar& lhs, const A& rhs)
{
	A result(rhs.shape());

	auto out = result.begin();
	for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(lhs << *it);
	}

	return result;
}

/**
 * @brief Right-shifts a scalar value by each array element as shift count.
 *
 * @param lhs Scalar value operand.
 * @param rhs Array shift-count operand.
 * @return Array with element-wise right-shift results.
 */
template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator>>(const Scalar& lhs, const A& rhs)
{
	A result(rhs.shape());

	auto out = result.begin();
	for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(lhs >> *it);
	}

	return result;
}

// In-place array-array

/**
 * @brief Applies element-wise bitwise AND assignment.
 *
 * @param lhs Left-hand operand to modify.
 * @param rhs Right-hand operand.
 * @return Reference to the modified left-hand operand.
 */
template<Array A>
requires Integer<typename A::value_type>
A& operator&=(A& lhs, const A& rhs)
{
	lhs = lhs & rhs;
	return lhs;
}

/**
 * @brief Applies element-wise bitwise OR assignment.
 *
 * @param lhs Left-hand operand to modify.
 * @param rhs Right-hand operand.
 * @return Reference to the modified left-hand operand.
 */
template<Array A>
requires Integer<typename A::value_type>
A& operator|=(A& lhs, const A& rhs)
{
	lhs = lhs | rhs;
	return lhs;
}

/**
 * @brief Applies element-wise bitwise XOR assignment.
 *
 * @param lhs Left-hand operand to modify.
 * @param rhs Right-hand operand.
 * @return Reference to the modified left-hand operand.
 */
template<Array A>
requires Integer<typename A::value_type>
A& operator^=(A& lhs, const A& rhs)
{
	lhs = lhs ^ rhs;
	return lhs;
}

/**
 * @brief Applies element-wise left-shift assignment.
 *
 * @param lhs Left-hand value operand to modify.
 * @param rhs Right-hand shift-count operand.
 * @return Reference to the modified left-hand operand.
 */
template<Array A>
requires Integer<typename A::value_type>
A& operator<<=(A& lhs, const A& rhs)
{
	lhs = lhs << rhs;
	return lhs;
}

/**
 * @brief Applies element-wise right-shift assignment.
 *
 * @param lhs Left-hand value operand to modify.
 * @param rhs Right-hand shift-count operand.
 * @return Reference to the modified left-hand operand.
 */
template<Array A>
requires Integer<typename A::value_type>
A& operator>>=(A& lhs, const A& rhs)
{
	lhs = lhs >> rhs;
	return lhs;
}

// In-place array-scalar

/**
 * @brief Applies bitwise AND assignment with a scalar.
 *
 * @param lhs Array operand to modify.
 * @param rhs Scalar operand.
 * @return Reference to the modified array.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator&=(A& lhs, const Scalar& rhs)
{
	lhs = lhs & rhs;
	return lhs;
}

/**
 * @brief Applies bitwise OR assignment with a scalar.
 *
 * @param lhs Array operand to modify.
 * @param rhs Scalar operand.
 * @return Reference to the modified array.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator|=(A& lhs, const Scalar& rhs)
{
	lhs = lhs | rhs;
	return lhs;
}

/**
 * @brief Applies bitwise XOR assignment with a scalar.
 *
 * @param lhs Array operand to modify.
 * @param rhs Scalar operand.
 * @return Reference to the modified array.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator^=(A& lhs, const Scalar& rhs)
{
	lhs = lhs ^ rhs;
	return lhs;
}

/**
 * @brief Applies left-shift assignment with a scalar shift count.
 *
 * @param lhs Array value operand to modify.
 * @param rhs Scalar shift count.
 * @return Reference to the modified array.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator<<=(A& lhs, const Scalar& rhs)
{
	lhs = lhs << rhs;
	return lhs;
}

/**
 * @brief Applies right-shift assignment with a scalar shift count.
 *
 * @param lhs Array value operand to modify.
 * @param rhs Scalar shift count.
 * @return Reference to the modified array.
 */
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator>>=(A& lhs, const Scalar& rhs)
{
	lhs = lhs >> rhs;
	return lhs;
}
