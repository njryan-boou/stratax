// TODO: Consider zero-copy reshape/flatten views once view support is implemented.

#pragma once

#include <algorithm>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

/** @brief Reshapes an array-like object into a tensor with the requested shape. */
template<Array A>
stratax::container::Tensor<typename A::value_type>
reshape(const A& arr, const stratax::core::Shape& shape)
{
	stratax::core::validation::require_equal_size(
		arr.size(),
		shape.elements(),
		"Reshape requires the new shape to contain the same number of elements.");

	stratax::container::Tensor<typename A::value_type> result(shape);
	std::copy(arr.begin(), arr.end(), result.begin());

	return result;
}

/** @brief Flattens an array-like object into a vector. */
template<Array A>
stratax::container::Vector<typename A::value_type> 
flatten(const A& arr)
{
	stratax::container::Vector<typename A::value_type> result(arr.size());
	std::copy(arr.begin(), arr.end(), result.begin());

	return result;
}
