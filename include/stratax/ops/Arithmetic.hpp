// TODO: Define result-container promotion rules for mixed container kinds.
#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/ops/Broadcasting.hpp>

#include <functional>

/**
 * @brief Applies an arithmetic callable to two broadcast-compatible arrays.
 *
 * The result preserves the common container kind, uses the promoted dtype of
 * the two operands, has their common broadcasted shape, and owns independent
 * storage. When @p check_zero_divisor is true, each used broadcasted
 * right-hand value is checked before invoking @p op.
 *
 * @tparam L Left Stratax array type and result container kind.
 * @tparam R Right Stratax array type with the same container kind as `L`.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Left array operand.
 * @param rhs Right array operand.
 * @param op Callable invoked for every broadcasted element pair.
 * @param check_zero_divisor Whether zero right-hand values are rejected.
 * @return Owning array rebound to the promoted operand dtype and containing
 *         the element-wise results.
 * @throws Exceptions::BroadcastError If the shapes are incompatible.
 * @throws Exceptions::ZeroDivisionError If divisor checking is enabled and a
 *         broadcasted right-hand element equals zero.
 * @throws Any exception propagated by allocation or @p op.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R, typename Op>
auto binary_op(
	const L& lhs,
	const R& rhs,
	Op op,
	bool check_zero_divisor = false)
{
	auto checked_op = [&](const auto& left, const auto& right)
	{
		if (check_zero_divisor &&
			right == typename R::value_type{})
		{
			throw Exceptions::ZeroDivisionError(
				"Array division divisor element cannot be zero.");
		}

		return op(left, right);
	};

	return broadcasted_op(lhs, rhs, checked_op);
}

/**
 * @brief Applies an arithmetic callable to an array and a right scalar.
 * The returned array preserves both the container type and dtype of @p lhs;
 * each callable result is converted to that dtype.
 *
 * @tparam A Stratax array and result type.
 * @tparam Scalar Numeric scalar type.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Array supplying each left argument.
 * @param rhs Scalar supplied as every right argument.
 * @param op Callable invoked for each array element.
 * @param check_zero_divisor Whether a zero @p rhs is rejected.
 * @return Owning `A` with the same shape as @p lhs.
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
 * The returned array preserves both the container type and dtype of @p rhs;
 * each callable result is converted to that dtype.
 *
 * @tparam Scalar Numeric scalar type.
 * @tparam A Stratax array and result type.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Scalar supplied as every left argument.
 * @param rhs Array supplying each right argument.
 * @param op Callable invoked for each array element.
 * @param check_zero_divisor Whether zero elements of @p rhs are rejected.
 * @return Owning `A` with the same shape as @p rhs.
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

/**
 * @brief Applies an arithmetic callable in place using array broadcasting.
 *
 * The right operand may broadcast to the existing shape of @p lhs, but the
 * operation never resizes or reshapes @p lhs. Callable results are converted
 * to `L::value_type` before assignment.
 *
 * @tparam L Mutable left Stratax array type.
 * @tparam R Right Stratax array type.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Array modified in place.
 * @param rhs Array broadcast over @p lhs.
 * @param op Callable invoked for every used element pair.
 * @param check_zero_divisor Whether used zero values from @p rhs are rejected.
 * @return Reference to @p lhs.
 * @throws Exceptions::BroadcastError If the shapes are incompatible or their
 *         broadcasted shape differs from the existing shape of @p lhs.
 * @throws Exceptions::ZeroDivisionError If divisor checking is enabled and a
 *         used right-hand element is zero.
 * @throws Any exception propagated by @p op.
 * @complexity O(n * r), where `n` is `lhs.size()` and `r` is its rank.
 */
template<Array L, Array R, typename Op>
L& compound_op(
	L& lhs,
	const R& rhs,
	Op op,
	bool check_zero_divisor = false)
{
	const auto result_shape =
		broadcasted_shape(lhs.shape(), rhs.shape());

	// Compound assignment cannot change the lhs shape.
	if (result_shape != lhs.shape())
	{
		throw Exceptions::BroadcastError(
			"Compound assignment cannot change the left-hand shape.");
	}

	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		const std::size_t rhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i,
				lhs.shape(),
				rhs.shape());

		if (check_zero_divisor &&
			rhs[rhs_index] == typename R::value_type{})
		{
			throw Exceptions::ZeroDivisionError(
				"Compound division divisor element cannot be zero.");
		}

		lhs[i] = static_cast<typename L::value_type>(
			op(lhs[i], rhs[rhs_index]));
	}

	return lhs;
}

/**
 * @brief Applies an arithmetic callable in place with a right scalar.
 *
 * Callable results are converted to `A::value_type` before assignment.
 *
 * @tparam A Mutable Stratax array type.
 * @tparam S Numeric scalar type.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Array modified in place.
 * @param rhs Scalar supplied as every right argument.
 * @param op Callable invoked for each element.
 * @param check_zero_divisor Whether a zero @p rhs is rejected.
 * @return Reference to @p lhs.
 * @throws Exceptions::ZeroDivisionError If divisor checking is enabled and
 *         @p rhs is zero, including when @p lhs is empty.
 * @throws Any exception propagated by @p op.
 * @complexity O(lhs.size()).
 */
template<Array A, Numeric S, typename Op>
A& compound_scalar_op(
	A& lhs,
	const S& rhs,
	Op op,
	bool check_zero_divisor = false)
{
	if (check_zero_divisor && rhs == S{})
	{
		throw Exceptions::ZeroDivisionError(
			"Compound division scalar divisor cannot be zero.");
	}

	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		lhs[i] = static_cast<typename A::value_type>(
			op(lhs[i], rhs));
	}

	return lhs;
}

/** @brief Adds two broadcast-compatible arrays. @return Owning broadcasted sum. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
auto operator+(const L& lhs, const R& rhs)
{
	return binary_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts two broadcast-compatible arrays. @return Owning `lhs - rhs` result. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
auto operator-(const L& lhs, const R& rhs)
{
	return binary_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies two broadcast-compatible arrays element-wise. @return Owning broadcasted product. @throws Exceptions::BroadcastError If shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
auto operator*(const L& lhs, const R& rhs)
{
	return binary_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides two broadcast-compatible arrays element-wise. @return Owning broadcasted quotient. @throws Exceptions::BroadcastError If shapes are incompatible. @throws Exceptions::ZeroDivisionError If a used divisor element is zero. @complexity O(n * r). */
template<Array L, Array R>
auto operator/(const L& lhs, const R& rhs)
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
auto operator-(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies every array element by a scalar. @return Owning result with the array shape. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
auto operator*(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides every array element by a scalar. @return Owning result with the array shape. @throws Exceptions::ZeroDivisionError If @p rhs is zero. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
auto operator/(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a scalar to every array element. @return Owning result with the array shape. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
auto operator+(const Scalar& lhs, const A& rhs)
{
	return rhs + lhs;
}

/** @brief Subtracts every array element from a scalar. @return Owning `lhs - rhs[i]` result. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
auto operator-(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies every array element by a scalar. @return Owning result with the array shape. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
auto operator*(const Scalar& lhs, const A& rhs)
{
	return rhs * lhs;
}

/** @brief Divides a scalar by every array element. @return Owning `lhs / rhs[i]` result. @throws Exceptions::ZeroDivisionError If any divisor element is zero. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
auto operator/(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a broadcast-compatible array in place. @return Reference to @p lhs; its shape is unchanged. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the shape of @p lhs. @complexity O(n * r). */
template<Array L, Array R>
L& operator+=(L& lhs, const R& rhs)
{
	return compound_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts a broadcast-compatible array in place. @return Reference to @p lhs; its shape is unchanged. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the shape of @p lhs. @complexity O(n * r). */
template<Array L, Array R>
L& operator-=(L& lhs, const R& rhs)
{
	return compound_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies by a broadcast-compatible array in place. @return Reference to @p lhs; its shape is unchanged. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the shape of @p lhs. @complexity O(n * r). */
template<Array L, Array R>
L& operator*=(L& lhs, const R& rhs)
{
	return compound_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides by a broadcast-compatible array in place. @return Reference to @p lhs; its shape is unchanged. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the shape of @p lhs. @throws Exceptions::ZeroDivisionError If a used divisor is zero. @complexity O(n * r). */
template<Array L, Array R>
L& operator/=(L& lhs, const R& rhs)
{
	return compound_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a scalar to every element in place. @return Reference to @p lhs. @complexity O(lhs.size()). */
template<Array A, Numeric S>
A& operator+=(A& lhs, const S& rhs)
{
	return compound_scalar_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts a scalar from every element in place. @return Reference to @p lhs. @complexity O(lhs.size()). */
template<Array A, Numeric S>
A& operator-=(A& lhs, const S& rhs)
{
	return compound_scalar_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies every element by a scalar in place. @return Reference to @p lhs. @complexity O(lhs.size()). */
template<Array A, Numeric S>
A& operator*=(A& lhs, const S& rhs)
{
	return compound_scalar_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides every element by a scalar in place. @return Reference to @p lhs. @throws Exceptions::ZeroDivisionError If @p rhs is zero, including for an empty array. @complexity O(lhs.size()). */
template<Array A, Numeric S>
A& operator/=(A& lhs, const S& rhs)
{
	return compound_scalar_op(lhs, rhs, std::divides<>{}, true);
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
