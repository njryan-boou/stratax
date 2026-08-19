#pragma once

#include <functional>

#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/core/ArrayTraits.hpp>
#include <stratax/core/dtypes/Types.hpp>
#include <stratax/ops/Broadcasting.hpp>

namespace stratax::core::comparison_detail {

/**
 * @brief Applies an element-wise comparison to two broadcast-compatible arrays.
 * @tparam L Left Stratax array type.
 * @tparam R Right Stratax array type.
 * @tparam Op Binary comparison callable.
 * @param lhs Left array operand.
 * @param rhs Right array operand.
 * @param op Comparison applied to each broadcasted element pair.
 * @return Owning boolean array whose container type is the promotion of @p lhs
 *         and @p rhs and whose shape is their common broadcasted shape.
 * @throws Exceptions::BroadcastError If the operand shapes are incompatible.
 * @complexity O(n * r), where `n` is the result size and `r` is its rank.
 * @internal
 */
template<Array L, Array R, typename Op>
auto comparison_op(
	const L& lhs,
	const R& rhs,
	Op op)
{
	using result_type =
		stratax::core::promote_array_t<
			L,
			R,
			stratax::dtype::bool_>;

	const auto result_shape =
		broadcasted_shape(lhs.shape(), rhs.shape());

	result_type result(result_shape);

	for (std::size_t i = 0; i < result.size(); ++i)
	{
		const std::size_t lhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i,
				result_shape,
				lhs.shape());

		const std::size_t rhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i,
				result_shape,
				rhs.shape());

		result[i] = static_cast<stratax::dtype::bool_>(
			op(lhs[lhs_index], rhs[rhs_index]));
	}

	return result;
}

/**
 * @brief Compares every array element with a scalar on the right.
 * @return Owning boolean array with @p lhs's container type and shape.
 * @complexity O(lhs.size()).
 * @internal
 */
template<Array A, DType Scalar, typename Op>
auto comparison_scalar_op(
	const A& lhs,
	const Scalar& rhs,
	Op op)
{
	using result_type =
		stratax::core::rebind_array_t<
			A,
			stratax::dtype::bool_>;

	result_type result(lhs.shape());

	auto out = result.begin();

	for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
	{
		*out = static_cast<stratax::dtype::bool_>(
			op(*it, rhs));
	}

	return result;
}

/**
 * @brief Compares a scalar on the left with every array element.
 * @return Owning boolean array with @p rhs's container type and shape.
 * @complexity O(rhs.size()).
 * @internal
 */
template<DType Scalar, Array A, typename Op>
auto comparison_scalar_op(
	const Scalar& lhs,
	const A& rhs,
	Op op)
{
	using result_type =
		stratax::core::rebind_array_t<
			A,
			stratax::dtype::bool_>;

	result_type result(rhs.shape());

	auto out = result.begin();

	for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
	{
		*out = static_cast<stratax::dtype::bool_>(
			op(lhs, *it));
	}

	return result;
}

/**
 * @brief Tests whether two arrays have identical shapes and element values.
 * @return `true` if the shapes and every corresponding value are equal.
 * @complexity O(r + n) in the worst case, where `r` is rank and `n` is size.
 * @internal
 */
template<Array L, Array R>
[[nodiscard]] bool array_equal(const L& lhs, const R& rhs)
{
	if (lhs.shape() != rhs.shape())
	{
		return false;
	}

	auto lhs_it = lhs.begin();
	auto rhs_it = rhs.begin();

	for (; lhs_it != lhs.end(); ++lhs_it, ++rhs_it)
	{
		if (*lhs_it != *rhs_it)
		{
			return false;
		}
	}

	return true;
}

} // namespace stratax::core::comparison_detail

/**
 * @brief Compares two arrays for element-wise equality with broadcasting.
 * @return Owning boolean array with the common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes cannot be broadcast.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R>
auto equal(const L& lhs, const R& rhs)
{
	return stratax::core::comparison_detail::comparison_op(
		lhs, rhs, std::equal_to<>{});
}

/**
 * @brief Compares two arrays for element-wise inequality with broadcasting.
 * @return Owning boolean array with the common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes cannot be broadcast.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R>
auto not_equal(const L& lhs, const R& rhs)
{
	return stratax::core::comparison_detail::comparison_op(
		lhs, rhs, std::not_equal_to<>{});
}

/**
 * @brief Tests each broadcasted left element for strict ordering below the
 *        corresponding right element.
 * @return Owning boolean array with the common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes cannot be broadcast.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R>
requires (
	Ordered<typename L::value_type> &&
	Ordered<typename R::value_type>
)
auto less(const L& lhs, const R& rhs)
{
	return stratax::core::comparison_detail::comparison_op(
		lhs, rhs, std::less<>{});
}

/**
 * @brief Tests each broadcasted left element for ordering below or equal to
 *        the corresponding right element.
 * @return Owning boolean array with the common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes cannot be broadcast.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R>
requires (
	Ordered<typename L::value_type> &&
	Ordered<typename R::value_type>
)
auto less_equal(const L& lhs, const R& rhs)
{
	return stratax::core::comparison_detail::comparison_op(
		lhs, rhs, std::less_equal<>{});
}

/**
 * @brief Tests each broadcasted left element for strict ordering above the
 *        corresponding right element.
 * @return Owning boolean array with the common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes cannot be broadcast.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R>
requires (
	Ordered<typename L::value_type> &&
	Ordered<typename R::value_type>
)
auto greater(const L& lhs, const R& rhs)
{
	return stratax::core::comparison_detail::comparison_op(
		lhs, rhs, std::greater<>{});
}

/**
 * @brief Tests each broadcasted left element for ordering above or equal to
 *        the corresponding right element.
 * @return Owning boolean array with the common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes cannot be broadcast.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R>
requires (
	Ordered<typename L::value_type> &&
	Ordered<typename R::value_type>
)
auto greater_equal(const L& lhs, const R& rhs)
{
	return stratax::core::comparison_detail::comparison_op(
		lhs, rhs, std::greater_equal<>{});
}

/**
 * @brief Compares every array element for equality with a right-hand scalar.
 * @return Owning boolean array with the same container and shape as @p lhs.
 * @complexity O(lhs.size()).
 */
template<Array A, DType Scalar>
auto equal(const A& lhs, const Scalar& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::equal_to<>{});
}

/**
 * @brief Compares a left-hand scalar for equality with every array element.
 * @return Owning boolean array with the same container and shape as @p rhs.
 * @complexity O(rhs.size()).
 */
template<DType Scalar, Array A>
auto equal(const Scalar& lhs, const A& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::equal_to<>{});
}

/**
 * @brief Compares every array element for inequality with a right-hand scalar.
 * @return Owning boolean array with the same container and shape as @p lhs.
 * @complexity O(lhs.size()).
 */
template<Array A, DType Scalar>
auto not_equal(const A& lhs, const Scalar& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::not_equal_to<>{});
}

/**
 * @brief Compares a left-hand scalar for inequality with every array element.
 * @return Owning boolean array with the same container and shape as @p rhs.
 * @complexity O(rhs.size()).
 */
template<DType Scalar, Array A>
auto not_equal(const Scalar& lhs, const A& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::not_equal_to<>{});
}

/**
 * @brief Tests each array element for strict ordering below a right-hand scalar.
 * @return Owning boolean array with the same container and shape as @p lhs.
 * @complexity O(lhs.size()).
 */
template<Array A, Ordered Scalar>
requires Ordered<typename A::value_type>
auto less(const A& lhs, const Scalar& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::less<>{});
}

/**
 * @brief Tests a left-hand scalar for strict ordering below each array element.
 * @return Owning boolean array with the same container and shape as @p rhs.
 * @complexity O(rhs.size()).
 */
template<Ordered Scalar, Array A>
requires Ordered<typename A::value_type>
auto less(const Scalar& lhs, const A& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::less<>{});
}

/**
 * @brief Tests each array element for ordering below or equal to a scalar.
 * @return Owning boolean array with the same container and shape as @p lhs.
 * @complexity O(lhs.size()).
 */
template<Array A, Ordered Scalar>
requires Ordered<typename A::value_type>
auto less_equal(const A& lhs, const Scalar& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::less_equal<>{});
}

/**
 * @brief Tests a scalar for ordering below or equal to each array element.
 * @return Owning boolean array with the same container and shape as @p rhs.
 * @complexity O(rhs.size()).
 */
template<Ordered Scalar, Array A>
requires Ordered<typename A::value_type>
auto less_equal(const Scalar& lhs, const A& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::less_equal<>{});
}

/**
 * @brief Tests each array element for strict ordering above a right-hand scalar.
 * @return Owning boolean array with the same container and shape as @p lhs.
 * @complexity O(lhs.size()).
 */
template<Array A, Ordered Scalar>
requires Ordered<typename A::value_type>
auto greater(const A& lhs, const Scalar& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::greater<>{});
}

/**
 * @brief Tests a left-hand scalar for strict ordering above each array element.
 * @return Owning boolean array with the same container and shape as @p rhs.
 * @complexity O(rhs.size()).
 */
template<Ordered Scalar, Array A>
requires Ordered<typename A::value_type>
auto greater(const Scalar& lhs, const A& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::greater<>{});
}

/**
 * @brief Tests each array element for ordering above or equal to a scalar.
 * @return Owning boolean array with the same container and shape as @p lhs.
 * @complexity O(lhs.size()).
 */
template<Array A, Ordered Scalar>
requires Ordered<typename A::value_type>
auto greater_equal(const A& lhs, const Scalar& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::greater_equal<>{});
}

/**
 * @brief Tests a scalar for ordering above or equal to each array element.
 * @return Owning boolean array with the same container and shape as @p rhs.
 * @complexity O(rhs.size()).
 */
template<Ordered Scalar, Array A>
requires Ordered<typename A::value_type>
auto greater_equal(const Scalar& lhs, const A& rhs)
{
	return stratax::core::comparison_detail::comparison_scalar_op(
		lhs, rhs, std::greater_equal<>{});
}

/**
 * @brief Element-wise array equality shorthand for equal().
 * @complexity O(n * r), including broadcast-index mapping.
 */
template<Array L, Array R>
auto operator==(const L& lhs, const R& rhs)
{
	return equal(lhs, rhs);
}

/**
 * @brief Element-wise array inequality shorthand for not_equal().
 * @complexity O(n * r), including broadcast-index mapping.
 */
template<Array L, Array R>
auto operator!=(const L& lhs, const R& rhs)
{
	return not_equal(lhs, rhs);
}

/**
 * @brief Element-wise array less-than shorthand for less().
 * @complexity O(n * r), including broadcast-index mapping.
 */
template<Array L, Array R>
requires (
	Ordered<typename L::value_type> &&
	Ordered<typename R::value_type>
)
auto operator<(const L& lhs, const R& rhs)
{
	return less(lhs, rhs);
}

/**
 * @brief Element-wise array less-than-or-equal shorthand for less_equal().
 * @complexity O(n * r), including broadcast-index mapping.
 */
template<Array L, Array R>
requires (
	Ordered<typename L::value_type> &&
	Ordered<typename R::value_type>
)
auto operator<=(const L& lhs, const R& rhs)
{
	return less_equal(lhs, rhs);
}

/**
 * @brief Element-wise array greater-than shorthand for greater().
 * @complexity O(n * r), including broadcast-index mapping.
 */
template<Array L, Array R>
requires (
	Ordered<typename L::value_type> &&
	Ordered<typename R::value_type>
)
auto operator>(const L& lhs, const R& rhs)
{
	return greater(lhs, rhs);
}

/**
 * @brief Element-wise array greater-than-or-equal shorthand for greater_equal().
 * @complexity O(n * r), including broadcast-index mapping.
 */
template<Array L, Array R>
requires (
	Ordered<typename L::value_type> &&
	Ordered<typename R::value_type>
)
auto operator>=(const L& lhs, const R& rhs)
{
	return greater_equal(lhs, rhs);
}

/**
 * @brief Element-wise array-to-scalar equality shorthand for equal().
 * @complexity O(lhs.size()).
 */
template<Array A, DType Scalar>
auto operator==(const A& lhs, const Scalar& rhs)
{
	return equal(lhs, rhs);
}

/**
 * @brief Element-wise scalar-to-array equality shorthand for equal().
 * @complexity O(rhs.size()).
 */
template<DType Scalar, Array A>
auto operator==(const Scalar& lhs, const A& rhs)
{
	return equal(lhs, rhs);
}

/**
 * @brief Element-wise array-to-scalar inequality shorthand for not_equal().
 * @complexity O(lhs.size()).
 */
template<Array A, DType Scalar>
auto operator!=(const A& lhs, const Scalar& rhs)
{
	return not_equal(lhs, rhs);
}

/**
 * @brief Element-wise scalar-to-array inequality shorthand for not_equal().
 * @complexity O(rhs.size()).
 */
template<DType Scalar, Array A>
auto operator!=(const Scalar& lhs, const A& rhs)
{
	return not_equal(lhs, rhs);
}

/**
 * @brief Element-wise array-to-scalar less-than shorthand for less().
 * @complexity O(lhs.size()).
 */
template<Array A, Ordered Scalar>
requires Ordered<typename A::value_type>
auto operator<(const A& lhs, const Scalar& rhs)
{
	return less(lhs, rhs);
}

/**
 * @brief Element-wise scalar-to-array less-than shorthand for less().
 * @complexity O(rhs.size()).
 */
template<Ordered Scalar, Array A>
requires Ordered<typename A::value_type>
auto operator<(const Scalar& lhs, const A& rhs)
{
	return less(lhs, rhs);
}

/**
 * @brief Element-wise array-to-scalar less-than-or-equal shorthand.
 * @complexity O(lhs.size()).
 */
template<Array A, Ordered Scalar>
requires Ordered<typename A::value_type>
auto operator<=(const A& lhs, const Scalar& rhs)
{
	return less_equal(lhs, rhs);
}

/**
 * @brief Element-wise scalar-to-array less-than-or-equal shorthand.
 * @complexity O(rhs.size()).
 */
template<Ordered Scalar, Array A>
requires Ordered<typename A::value_type>
auto operator<=(const Scalar& lhs, const A& rhs)
{
	return less_equal(lhs, rhs);
}

/**
 * @brief Element-wise array-to-scalar greater-than shorthand for greater().
 * @complexity O(lhs.size()).
 */
template<Array A, Ordered Scalar>
requires Ordered<typename A::value_type>
auto operator>(const A& lhs, const Scalar& rhs)
{
	return greater(lhs, rhs);
}

/**
 * @brief Element-wise scalar-to-array greater-than shorthand for greater().
 * @complexity O(rhs.size()).
 */
template<Ordered Scalar, Array A>
requires Ordered<typename A::value_type>
auto operator>(const Scalar& lhs, const A& rhs)
{
	return greater(lhs, rhs);
}

/**
 * @brief Element-wise array-to-scalar greater-than-or-equal shorthand.
 * @complexity O(lhs.size()).
 */
template<Array A, Ordered Scalar>
requires Ordered<typename A::value_type>
auto operator>=(const A& lhs, const Scalar& rhs)
{
	return greater_equal(lhs, rhs);
}

/**
 * @brief Element-wise scalar-to-array greater-than-or-equal shorthand.
 * @complexity O(rhs.size()).
 */
template<Ordered Scalar, Array A>
requires Ordered<typename A::value_type>
auto operator>=(const Scalar& lhs, const A& rhs)
{
	return greater_equal(lhs, rhs);
}
