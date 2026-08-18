// TODO: Consider zero-copy reshape/flatten views once view support is implemented.

#pragma once

#include <algorithm>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

namespace stratax::manipulation {

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

template<Array A>
[[nodiscard]]
stratax::container::Vector<typename A::value_type> 
flatten(const A& arr)
{
	stratax::container::Vector<typename A::value_type> result(arr.size());
	std::copy(arr.begin(), arr.end(), result.begin());

	return result;
}

}
