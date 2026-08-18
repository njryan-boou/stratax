// TODO: Add result-container and dtype promotion for broadcasted operations.
// Mixed container types may require promotion to Tensor, and the result
// value type should be derived from the operation rather than the left operand.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>

namespace stratax::core::broadcast_detail {

/**
 * @brief Returns a shape dimension counted from the trailing axis.
 *
 * Missing leading dimensions are treated as singleton dimensions, which is
 * equivalent to left-padding a lower-rank shape with ones for broadcasting.
 *
 * @param shape Shape to query.
 * @param offset Zero-based distance from the final dimension.
 * @return The selected dimension, or one when @p offset exceeds the rank.
 * @complexity O(1).
 * @internal
 */
inline std::size_t dimension_from_right(
	const Shape& shape,
	std::size_t offset)
{
	return offset < shape.rank()
		? shape[shape.rank() - 1 - offset]
		: 1;
}

/**
 * @brief Maps a flat result index to the corresponding operand index.
 *
 * Coordinates are decoded from @p result_shape in row-major order. Coordinates
 * on singleton operand dimensions collapse to zero, while missing leading
 * operand dimensions are ignored.
 *
 * @param result_index Flat row-major index in the broadcasted result.
 * @param result_shape Shape of the broadcasted result.
 * @param operand_shape Broadcast-compatible shape of one operand.
 * @return Flat row-major index of the operand value used at @p result_index.
 * @pre @p result_index is in range, every result dimension is nonzero, and
 *      @p operand_shape is broadcast-compatible with @p result_shape.
 * @complexity O(result_shape.rank()).
 * @internal
 */
inline std::size_t flat_operand_index(
	std::size_t result_index,
	const Shape& result_shape,
	const Shape& operand_shape)
{
	std::size_t operand_index = 0;
	std::size_t operand_stride = 1;

	for (std::size_t result_axis = result_shape.rank(); result_axis-- > 0;)
	{
		const std::size_t coordinate =
			result_index % result_shape[result_axis];
		result_index /= result_shape[result_axis];

		const std::size_t offset = result_shape.rank() - 1 - result_axis;
		if (offset >= operand_shape.rank())
		{
			continue;
		}

		const std::size_t operand_axis = operand_shape.rank() - 1 - offset;
		const std::size_t operand_dimension = operand_shape[operand_axis];
		const std::size_t operand_coordinate =
			operand_dimension == 1 ? 0 : coordinate;

		operand_index += operand_coordinate * operand_stride;
		operand_stride *= operand_dimension;
	}

	return operand_index;
}

} // namespace stratax::core::broadcast_detail

/**
 * @brief Tests whether two shapes satisfy trailing-dimension broadcasting.
 *
 * Shapes are aligned from the right. Each aligned pair is compatible when the
 * dimensions are equal or either dimension is one. Missing leading dimensions
 * behave as though they were one. A zero dimension therefore broadcasts with
 * zero or one, but not with a different non-singleton extent.
 *
 * @param shape1 First candidate shape.
 * @param shape2 Second candidate shape.
 * @return `true` when the shapes can be broadcast together; otherwise `false`.
 * @complexity O(max(shape1.rank(), shape2.rank())).
 */
inline bool broadcastable(
	const stratax::core::Shape& shape1,
	const stratax::core::Shape& shape2)
{
	const std::size_t result_rank = std::max(shape1.rank(), shape2.rank());

	for (std::size_t offset = 0; offset < result_rank; ++offset)
	{
		const std::size_t left =
			stratax::core::broadcast_detail::dimension_from_right(shape1, offset);
		const std::size_t right =
			stratax::core::broadcast_detail::dimension_from_right(shape2, offset);

		if (left != right && left != 1 && right != 1)
		{
			return false;
		}
	}

	return true;
}

/**
 * @brief Computes the common shape produced by broadcasting two shapes.
 *
 * Compatible dimensions are aligned from the right and the non-singleton
 * extent is selected for each result axis. The operation is symmetric: swapping
 * the arguments produces the same shape.
 *
 * @param shape1 First operand shape.
 * @param shape2 Second operand shape.
 * @return Common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes are incompatible.
 * @throws std::bad_alloc If result-dimension storage cannot be allocated.
 * @complexity O(max(shape1.rank(), shape2.rank())).
 */
inline stratax::core::Shape broadcasted_shape(
	const stratax::core::Shape& shape1,
	const stratax::core::Shape& shape2)
{
	if (!broadcastable(shape1, shape2))
	{
		throw Exceptions::BroadcastError("Shapes are not broadcastable");
	}

	const std::size_t result_rank = std::max(shape1.rank(), shape2.rank());
	std::vector<std::size_t> result(result_rank);

	for (std::size_t offset = 0; offset < result_rank; ++offset)
	{
		const std::size_t left =
			stratax::core::broadcast_detail::dimension_from_right(shape1, offset);
		const std::size_t right =
			stratax::core::broadcast_detail::dimension_from_right(shape2, offset);

		result[result_rank - 1 - offset] = left == 1 ? right : left;
	}

	return stratax::core::Shape{result};
}

// TODO: Define result-container promotion rules for mixed container types.
// Same-type operations should preserve the container type;
// mixed Vector/Matrix/Tensor operations may promote to Tensor.
/**
 * @brief Applies a binary callable element-wise using array broadcasting.
 *
 * The result uses the left operand's container and value type. Each flat result
 * coordinate is mapped independently into both operands, collapsing broadcast
 * dimensions to coordinate zero. The returned container owns new storage.
 * Current callers must ensure the broadcasted shape is valid for `L`; mixed
 * container promotion and operation-result type deduction are not yet provided.
 *
 * @tparam L Left Stratax array type and result container type.
 * @tparam R Right Stratax array type.
 * @tparam Op Binary callable whose result is assignable to `L::value_type`.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @param op Callable invoked as `op(lhs_value, rhs_value)`.
 * @return Owning `L` with the broadcasted shape and computed values.
 * @throws Exceptions::BroadcastError If the operand shapes are incompatible.
 * @throws Any exception propagated by result construction or @p op.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R, typename Op>
auto broadcasted_op(const L& lhs, const R& rhs, Op op)
{

	const auto result_shape =
    	broadcasted_shape(lhs.shape(), rhs.shape());
	L result(result_shape);

	for (std::size_t i = 0; i < result.size(); ++i)
	{
		const std::size_t lhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i, result_shape, lhs.shape());
		const std::size_t rhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i, result_shape, rhs.shape());

		result[i] = op(lhs[lhs_index], rhs[rhs_index]);
	}

	return result;
}

/**
 * @brief Applies a binary callable to every array element and a right scalar.
 * @tparam L Stratax array type and result container type.
 * @tparam S Numeric scalar type.
 * @tparam Op Binary callable whose result is assignable to `L::value_type`.
 * @param lhs Array supplying each left argument.
 * @param rhs Scalar supplied as every right argument.
 * @param op Callable invoked as `op(lhs[i], rhs)`.
 * @return Owning `L` with the same shape as @p lhs and computed values.
 * @throws Any exception propagated by result construction or @p op.
 * @complexity O(lhs.size()).
 */
template<Array L, Numeric S, typename Op>
L broadcasted_op(const L& lhs, const S& rhs, Op op)
{
	L result(lhs.shape());

	for (std::size_t i = 0; i < result.size(); ++i)
	{
		result[i] = op(lhs[i], rhs);
	}

	return result;
}

/**
 * @brief Applies a binary callable to a left scalar and every array element.
 * @tparam S Numeric scalar type.
 * @tparam R Stratax array type and result container type.
 * @tparam Op Binary callable whose result is assignable to `R::value_type`.
 * @param lhs Scalar supplied as every left argument.
 * @param rhs Array supplying each right argument.
 * @param op Callable invoked as `op(lhs, rhs[i])`.
 * @return Owning `R` with the same shape as @p rhs and computed values.
 * @throws Any exception propagated by result construction or @p op.
 * @complexity O(rhs.size()).
 */
template<Numeric S, Array R, typename Op>
R broadcasted_op(const S& lhs, const R& rhs, Op op)
{
	R result(rhs.shape());

	for (std::size_t i = 0; i < result.size(); ++i)
	{
		result[i] = op(lhs, rhs[i]);
	}

	return result;
}
