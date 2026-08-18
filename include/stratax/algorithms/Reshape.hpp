// TODO: Consider zero-copy reshape/flatten views once view support is implemented.

#pragma once

#include <algorithm>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

namespace stratax::manipulation {

/**
 * @brief Copies an owning Stratax array into a tensor with a new shape.
 *
 * The target shape must describe exactly the same number of elements as the
 * source. Values are copied in flat iterator order, so only the logical shape
 * and row-major strides change. The returned tensor owns independent storage.
 *
 * @tparam A Vector, Matrix, or Tensor type satisfying Array.
 * @param arr Source array whose flat element order is preserved.
 * @param shape Requested shape of the returned tensor.
 * @return Owning Tensor with `A::value_type`, @p shape, and copied values.
 * @throws Exceptions::DimensionError If `shape.elements()` or a target stride
 *         cannot be represented.
 * @throws Exceptions::ShapeError If `shape.elements() != arr.size()`.
 * @throws std::bad_alloc If output storage or metadata allocation fails.
 * @complexity O(arr.size() + shape.rank()).
 */
template<Array A>
[[nodiscard]]
stratax::container::Tensor<typename A::value_type>
reshape(const A& arr, const stratax::core::Shape& shape)
{
	if (arr.size() != shape.elements())
	{
		throw Exceptions::ShapeError(
			"Reshape size must match original array size.");
	}

	stratax::container::Tensor<typename A::value_type> result(shape);
	std::copy(arr.begin(), arr.end(), result.begin());

	return result;
}

/**
 * @brief Copies an owning Stratax array into a rank-one vector.
 *
 * Values retain the source array's flat row-major iterator order. The returned
 * vector has shape `{arr.size()}` and owns storage independent of @p arr.
 *
 * @tparam A Vector, Matrix, or Tensor type satisfying Array.
 * @param arr Source array to flatten.
 * @return Owning Vector with `A::value_type` and copied values.
 * @throws std::bad_alloc If output allocation fails.
 * @complexity O(arr.size()).
 */
template<Array A>
[[nodiscard]]
stratax::container::Vector<typename A::value_type>
flatten(const A& arr)
{
	stratax::container::Vector<typename A::value_type> result(arr.size());
	std::copy(arr.begin(), arr.end(), result.begin());

	return result;
}

} // namespace stratax::manipulation
