#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>

namespace stratax::core::broadcast_detail {

inline std::size_t dimension_from_right(
	const Shape& shape,
	std::size_t offset)
{
	return offset < shape.rank()
		? shape[shape.rank() - 1 - offset]
		: 1;
}

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