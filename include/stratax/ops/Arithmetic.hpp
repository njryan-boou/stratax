/** @file
 * @brief Element-wise arithmetic, scalar promotion, and compound assignment.
 *
 * Operations that allocate results require owning-container result trait
 * specializations. Element operators receive the original operand types; the
 * result is converted afterward. Native C++ arithmetic and conversion rules
 * apply, including representability requirements. Division rejects zero and
 * signed expression-type minimum divided by minus one. Compound operations
 * use the operands' initial values, including overlapping views; repeated
 * destination offsets receive the last logical write. Operations involving
 * views stage O(n) results. Allocation failures propagate.
 */
#pragma once

#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/ops/Broadcasting.hpp>
#include <stratax/ops/Compound.hpp>
#include <stratax/ops/Numeric.hpp>

#include <functional>

/**
 * @brief Applies an arithmetic callable to two broadcast-compatible arrays.
 *
 * Matching container kinds are preserved; mixed kinds produce Tensor. The result
 * uses the promoted dtype of
 * the two operands, has their common broadcasted shape, and owns independent
 * storage. When @p check_zero_divisor is true, each used broadcasted
 * right-hand value is checked before invoking @p op.
 *
 * @tparam L Left owning Stratax array type.
 * @tparam R Right owning Stratax array type; it may differ from `L`.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Left array operand.
 * @param rhs Right array operand.
 * @param op Callable invoked for every broadcasted element pair.
 * @param check_zero_divisor Whether zero right-hand values are rejected.
 * @return Owning array rebound to the promoted operand dtype and containing
 *         the element-wise results.
 * @throws Exceptions::BroadcastError If shapes are incompatible or an empty operand would supply values to a nonempty result.
 * @throws Exceptions::ZeroDivisionError If divisor checking is enabled and a
 *         broadcasted right-hand element equals zero.
 * @throws Exceptions::OverflowError If divisor checking is enabled and signed division is not representable.
 * @throws Any exception propagated by allocation or @p op.
 * @complexity O((n + 1) * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R, typename Op>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
auto binary_op(
	const L& lhs,
	const R& rhs,
	Op op,
	bool check_zero_divisor = false)
{
	auto checked_op = [&](const auto& left, const auto& right)
	{
		if (check_zero_divisor)
		{
			stratax::core::numeric_detail::require_valid_division(left, right);
		}

		return op(left, right);
	};

	using result_value_type =
		stratax::core::promote_t<
			typename L::value_type,
			typename R::value_type>;

	return stratax::core::broadcasted_op<result_value_type>(
		lhs,
		rhs,
		checked_op);
}

/**
 * @brief Applies an arithmetic callable to an array and a right scalar.
 * The returned array preserves the container family and shape of @p lhs.
 * Its dtype is selected by the array/scalar promotion rules.
 *
 * @tparam A Stratax array and result type.
 * @tparam Scalar Numeric scalar type.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Array supplying each left argument.
 * @param rhs Scalar supplied as every right argument.
 * @param op Callable invoked for each array element.
 * @param check_zero_divisor Whether a zero @p rhs is rejected.
 * @return Owning array with the same shape and container family as @p lhs, and promoted dtype.
 * @throws Exceptions::ZeroDivisionError If divisor checking is enabled and
 *         @p rhs equals zero, including when @p lhs is empty.
 * @throws Exceptions::OverflowError If divisor checking is enabled and signed division is not representable.
 * @throws Any exception propagated by allocation or @p op.
 * @complexity O(lhs.size()).
 */
template<Array A, Numeric Scalar, typename Op>
auto binary_scalar_op(const A& lhs, const Scalar& rhs, Op op, bool check_zero_divisor = false)
{
	if (check_zero_divisor && rhs == Scalar{})
	{
		throw Exceptions::ZeroDivisionError("Division by zero.");
	}

	auto checked_op = [&](const auto& left, const auto& right)
	{
		if (check_zero_divisor)
		{
			stratax::core::numeric_detail::require_valid_division(left, right);
		}
		return op(left, right);
	};
	return broadcasted_op(lhs, rhs, checked_op);
}

/**
 * @brief Applies an arithmetic callable to a left scalar and an array.
 * The returned array preserves the container family and shape of @p rhs.
 * Its dtype is selected by the array/scalar promotion rules.
 *
 * @tparam Scalar Numeric scalar type.
 * @tparam A Stratax array and result type.
 * @tparam Op Binary arithmetic callable.
 * @param lhs Scalar supplied as every left argument.
 * @param rhs Array supplying each right argument.
 * @param op Callable invoked for each array element.
 * @param check_zero_divisor Whether zero elements of @p rhs are rejected.
 * @return Owning array with the same shape and container family as @p rhs, and promoted dtype.
 * @throws Exceptions::ZeroDivisionError If divisor checking is enabled and an
 *         element of @p rhs equals zero.
 * @throws Exceptions::OverflowError If divisor checking is enabled and signed division is not representable.
 * @throws Any exception propagated by allocation or @p op.
 * @complexity O(rhs.size()).
 */
template<Numeric Scalar, Array A, typename Op>
auto binary_scalar_op(const Scalar& lhs, const A& rhs, Op op, bool check_zero_divisor = false)
{
	auto checked_op = [&](const auto& left, const auto& right)
	{
		if (check_zero_divisor)
		{
			stratax::core::numeric_detail::require_valid_division(left, right);
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
 * to `L::value_type` before assignment. All built-in validation precedes writes.
 * Operands are read as they were before mutation; if destination elements alias,
 * the last logical write wins. Non-owning operands require O(n) staging storage.
 * A throwing callable may leave partial writes when both operands own storage.
 * @pre The callable does not modify either operand or its metadata.
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
 * @throws Exceptions::OverflowError If divisor checking is enabled and signed division is not representable.
 * @throws Any exception propagated by allocation or @p op.
 * @complexity O((n + 1) * r), where `n` is `lhs.size()` and `r` is its rank.
 */
template<Array L, Array R, typename Op>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
L& compound_op(
	L& lhs,
	const R& rhs,
	Op op,
	bool check_zero_divisor = false)
{
	const auto result_shape =
		stratax::core::broadcast_detail::array_result_shape(lhs, rhs);

	// Compound assignment cannot change the lhs shape.
	if (result_shape != lhs.shape())
	{
		throw Exceptions::BroadcastError(
			"In-place broadcasting cannot change the left operand's shape.");
	}

	// Validate all used divisors before changing any elements.
	if (check_zero_divisor)
	{
		for (std::size_t i = 0; i < lhs.size(); ++i)
		{
			const auto rhs_index = stratax::core::broadcast_detail::flat_operand_index(
				i, lhs.shape(), rhs.shape());
			stratax::core::numeric_detail::require_valid_division(lhs[i], rhs[rhs_index]);
		}
	}

	constexpr bool stage = !stratax::core::compound_detail::owns_storage<L> ||
		!stratax::core::compound_detail::owns_storage<R>;
	return stratax::core::compound_detail::write_results<stage>(lhs, [&](std::size_t i)
	{
		const auto rhs_index = stratax::core::broadcast_detail::flat_operand_index(
			i, lhs.shape(), rhs.shape());
		return op(lhs[i], rhs[rhs_index]);
	});
}

/**
 * @brief Applies an arithmetic callable in place with a right scalar.
 *
 * Callable results are converted to `A::value_type` before assignment. The
 * scalar is copied first, even when it references a destination element.
 * Views stage O(n) results; repeated destination offsets use the last result.
 * A throwing callable may leave partial writes for owning destinations.
 * @pre The callable does not modify either operand or its metadata.
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
 * @throws Exceptions::OverflowError If divisor checking is enabled and signed division is not representable.
 * @throws Any exception propagated by allocation or @p op.
 * @complexity O(n) for owning arrays, O((n + 1) * r) for views.
 */
template<Array A, Numeric S, typename Op>
A& compound_scalar_op(
	A& lhs,
	const S& rhs,
	Op op,
	bool check_zero_divisor = false)
{
	const S scalar = rhs;
	if (check_zero_divisor)
	{
		if (scalar == S{})
		{
			throw Exceptions::ZeroDivisionError("Division by zero.");
		}
		for (std::size_t i = 0; i < lhs.size(); ++i)
		{
			stratax::core::numeric_detail::require_valid_division(lhs[i], scalar);
		}
	}

	return stratax::core::compound_detail::write_results<
		!stratax::core::compound_detail::owns_storage<A>>(lhs, [&](std::size_t i)
	{
		return op(lhs[i], scalar);
	});
}

/** @brief Adds two broadcast-compatible arrays. @return Owning broadcasted sum. @throws Exceptions::BroadcastError If shapes are incompatible or an empty operand would supply values to a nonempty result. @complexity O((n + 1) * r). */
template<Array L, Array R>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
auto operator+(const L& lhs, const R& rhs)
{
	return binary_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts two broadcast-compatible arrays. @return Owning `lhs - rhs` result. @throws Exceptions::BroadcastError If shapes are incompatible or an empty operand would supply values to a nonempty result. @complexity O((n + 1) * r). */
template<Array L, Array R>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
auto operator-(const L& lhs, const R& rhs)
{
	return binary_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies two broadcast-compatible arrays element-wise. @return Owning broadcasted product. @throws Exceptions::BroadcastError If shapes are incompatible or an empty operand would supply values to a nonempty result. @complexity O((n + 1) * r). */
template<Array L, Array R>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
auto operator*(const L& lhs, const R& rhs)
{
	return binary_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides two broadcast-compatible arrays element-wise. @return Owning broadcasted quotient. @throws Exceptions::BroadcastError If shapes are incompatible or an empty operand would supply values to a nonempty result. @throws Exceptions::ZeroDivisionError If a used divisor element is zero. @throws Exceptions::OverflowError If native signed division is not representable. @complexity O((n + 1) * r). */
template<Array L, Array R>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
auto operator/(const L& lhs, const R& rhs)
{
	return binary_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a scalar to every array element. @return Owning result with the array shape. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
requires Numeric<typename A::value_type>
auto operator+(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts a scalar from every array element. @return Owning result with the array shape. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
requires Numeric<typename A::value_type>
auto operator-(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies every array element by a scalar. @return Owning result with the array shape. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
requires Numeric<typename A::value_type>
auto operator*(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides every array element by a scalar. @return Owning result with the array shape. @throws Exceptions::ZeroDivisionError If @p rhs is zero. @throws Exceptions::OverflowError If native signed division is not representable. @complexity O(lhs.size()). */
template<Array A, Numeric Scalar>
requires Numeric<typename A::value_type>
auto operator/(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a scalar to every array element. @return Owning result with the array shape. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
requires Numeric<typename A::value_type>
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
requires Numeric<typename A::value_type>
auto operator*(const Scalar& lhs, const A& rhs)
{
	return rhs * lhs;
}

/** @brief Divides a scalar by every array element. @return Owning `lhs / rhs[i]` result. @throws Exceptions::ZeroDivisionError If any divisor element is zero. @throws Exceptions::OverflowError If native signed division is not representable. @complexity O(rhs.size()). */
template<Numeric Scalar, Array A>
requires Numeric<typename A::value_type>
auto operator/(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a broadcast-compatible array in place. @return Reference to @p lhs; its shape is unchanged. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the shape of @p lhs. @complexity O((n + 1) * r). */
template<Array L, Array R>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
L& operator+=(L& lhs, const R& rhs)
{
	return compound_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts a broadcast-compatible array in place. @return Reference to @p lhs; its shape is unchanged. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the shape of @p lhs. @complexity O((n + 1) * r). */
template<Array L, Array R>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
L& operator-=(L& lhs, const R& rhs)
{
	return compound_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies by a broadcast-compatible array in place. @return Reference to @p lhs; its shape is unchanged. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the shape of @p lhs. @complexity O((n + 1) * r). */
template<Array L, Array R>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
L& operator*=(L& lhs, const R& rhs)
{
	return compound_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides by a broadcast-compatible array in place. @return Reference to @p lhs; its shape is unchanged. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the shape of @p lhs. @throws Exceptions::ZeroDivisionError If a used divisor is zero. @throws Exceptions::OverflowError If native signed division is not representable. @complexity O((n + 1) * r). */
template<Array L, Array R>
requires (
	Numeric<typename L::value_type> &&
	Numeric<typename R::value_type>
)
L& operator/=(L& lhs, const R& rhs)
{
	return compound_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Adds a scalar to every element in place. @return Reference to @p lhs. @complexity O(lhs.size()). */
template<Array A, Numeric S>
requires Numeric<typename A::value_type>
A& operator+=(A& lhs, const S& rhs)
{
	return compound_scalar_op(lhs, rhs, std::plus<>{});
}

/** @brief Subtracts a scalar from every element in place. @return Reference to @p lhs. @complexity O(lhs.size()). */
template<Array A, Numeric S>
requires Numeric<typename A::value_type>
A& operator-=(A& lhs, const S& rhs)
{
	return compound_scalar_op(lhs, rhs, std::minus<>{});
}

/** @brief Multiplies every element by a scalar in place. @return Reference to @p lhs. @complexity O(lhs.size()). */
template<Array A, Numeric S>
requires Numeric<typename A::value_type>
A& operator*=(A& lhs, const S& rhs)
{
	return compound_scalar_op(lhs, rhs, std::multiplies<>{});
}

/** @brief Divides every element by a scalar in place. @return Reference to @p lhs. @throws Exceptions::ZeroDivisionError If @p rhs is zero, including for an empty array. @throws Exceptions::OverflowError If native signed division is not representable. @complexity O(n) for owning arrays; O((n + 1) * r) for views. */
template<Array A, Numeric S>
requires Numeric<typename A::value_type>
A& operator/=(A& lhs, const S& rhs)
{
	return compound_scalar_op(lhs, rhs, std::divides<>{}, true);
}

/** @brief Returns an owning element-wise negation with the input shape and dtype.
 * @tparam A Owning numeric array whose value type can be list-initialized from -1.
 * @note Unsigned dtypes are not supported by this implementation; native arithmetic representability requirements apply.
 * @complexity O(arr.size() + arr.rank()).
 */
template<Array A>
requires Numeric<typename A::value_type>
A operator-(const A& arr)
{
	return arr * typename A::value_type{-1};
}

/** @brief Returns a copy of the array object with unchanged values.
 * @note Owning arrays copy their storage; copying an ArrayView preserves aliasing.
 * @complexity O(arr.size() + arr.rank()) for owning arrays; O(arr.rank()) for views.
 */
template<Array A>
requires Numeric<typename A::value_type>
A operator+(const A& arr)
{
	return arr;
}
