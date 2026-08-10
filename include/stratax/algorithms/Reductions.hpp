#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/indexing/Slicing.hpp>
#include <stratax/algorithms/Conversion.hpp>

#include <numeric>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

namespace reduction {

/** @brief Advances a row-major multidimensional index. */
inline bool advance(const stratax::core::Shape& shape, std::vector<std::size_t>& indices)
{
	// Start from the rightmost dimension
	for (int d = shape.rank() - 1; d >= 0; --d) {
		indices[d]++;
		if (indices[d] < shape(d)) {
			return true;  // Successfully advanced, more indices exist
		}
		indices[d] = 0;  // Reset and carry over to next dimension
	}
    
	return false;  // All dimensions wrapped around, no more indices
}

/** @brief Converts a possibly negative axis into a non-negative axis. */
template<Array A>
inline int normalize_axis(const A& arr, int axis)
{
	int init = axis;
	if (axis < 0)
	{
		init += arr.rank();
	}

	return init;
}

/** @brief Computes the shape produced by an axis reduction. */
template<Array A>
inline std::vector<std::size_t> result_shape(const A& arr, int axis, bool keepdims)
{
	stratax::core::Shape input_shape = arr.shape();

	std::vector<std::size_t> result_dimensions;
    
	if (keepdims)
	{
		for (std::size_t dimension = 0; dimension < input_shape.rank(); ++dimension)
		{
			if (dimension == static_cast<std::size_t>(axis))
			{
				result_dimensions.push_back(1);
			}

			else
			{
				result_dimensions.push_back(input_shape[dimension]);
			}
		}
	}

	else 
	{
		for (std::size_t dimension = 0; dimension < input_shape.rank(); ++dimension)
		{
			if (dimension != static_cast<std::size_t>(axis))
			{
				result_dimensions.push_back(input_shape[dimension]);
			}
		}
	}

	return result_dimensions;
}

/** @brief Result value type returned by an axis reduction callback. */
template<Array A, typename Func>
using axis_reduce_value_t =
	decltype(std::declval<Func>()(
		std::declval<const stratax::container::Tensor<typename A::value_type>&>()));

/** @brief Applies a scalar reduction callback along one axis. */
template<Array A, typename Func>
stratax::container::Tensor<axis_reduce_value_t<A, Func>>
axis_reduce(const A& array, int axis, Func func, bool keepdims = false)
{
	using ResultType = axis_reduce_value_t<A, Func>;

	int Axis = normalize_axis(array, axis);
    
	if (Axis < 0 || Axis >= static_cast<int>(array.rank()))
	{
		throw Exceptions::AxisError("axis is out of range.");
	}

	stratax::container::Tensor<typename A::value_type> arr = to_tensor(array);

	const stratax::core::Shape input_shape = arr.shape();

	std::vector<std::size_t> result_dims = result_shape(array, Axis, keepdims);

	// A zero-dimensional tensor cannot store values in the current API.
	// Represent scalar reductions as a single-element tensor.
	if (result_dims.empty())
	{
		ResultType scalar_result = func(arr);
		return stratax::container::Tensor<ResultType>(stratax::core::Shape{1}, scalar_result);
	}

	stratax::container::Tensor<ResultType> result(stratax::core::Shape{result_dims});

	std::vector<std::size_t> output_index(result.rank(), 0);
	std::vector<stratax::core::Slice> slices;
    
	do {
		slices.clear();
		std::size_t output_position = 0;
		for (std::size_t dimension = 0; dimension < input_shape.rank(); ++dimension)
		{
			if (static_cast<int>(dimension) == Axis)
			{
				slices.push_back(stratax::core::Slice{static_cast<std::ptrdiff_t>(0), 
				static_cast<std::ptrdiff_t>(input_shape[dimension])});
			}

			else
			{
				const std::size_t index = keepdims
					? output_index[dimension]
					: output_index[output_position++];

				slices.push_back(stratax::core::Slice{
					static_cast<std::ptrdiff_t>(index),
					static_cast<std::ptrdiff_t>(index + 1)
				});
			}
		}
		auto s = slice(arr, slices);
		ResultType value = func(s);
		result(output_index) = value;
	}
		while (reduction::advance(result.shape(), output_index));

	return result;
}

// Global Reductions

/** @brief Returns the sum of all elements. */
template<Array A>
typename A::value_type sum(const A& arr)
{
	return std::accumulate(
		arr.begin(),
		arr.end(),
		typename A::value_type(0)
	);
}

/** @brief Returns the product of all elements. */
template<Array A>
typename A::value_type prod(const A& arr)
{
	return std::accumulate(
		arr.begin(),
		arr.end(),
		typename A::value_type(1),
		std::multiplies<typename A::value_type>()
	);
}

/** @brief Returns the largest element. */
template<Array A>
typename A::value_type max(const A& arr)
{
	auto result = std::max_element(
		arr.begin(),
		arr.end()
	);

	return *result;
}

/** @brief Returns the smallest element. */
template<Array A>
typename A::value_type min(const A& arr)
{
	auto result = std::min_element(
		arr.begin(),
		arr.end()
	);

	return *result;
}

/** @brief Returns the flat index of the largest element. */
template<Array A>
std::size_t argmax(const A& arr)
{
	auto result = std::max_element(
		arr.begin(),
		arr.end()
	);

	return static_cast<std::size_t>(std::distance(arr.begin(), result));
}

/** @brief Returns the flat index of the smallest element. */
template<Array A>
std::size_t argmin(const A& arr)
{
	auto result = std::min_element(
		arr.begin(),
		arr.end()
	);

	return static_cast<std::size_t>(std::distance(arr.begin(), result));
}

/** @brief Returns the arithmetic mean of all elements. */
template<Array A>
double mean(const A& arr)
{
	return static_cast<double>(sum(arr)) / static_cast<double>(arr.size());
}

/** @brief Returns the population variance of all elements. */
template<Array A>
double var(const A& arr)
{
	double count = 0.0;
	double mean_value = 0.0;
	double m2 = 0.0;

	for (const auto& value : arr)
	{
		count += 1.0;
		const double delta = static_cast<double>(value) - mean_value;
		mean_value += delta / count;
		const double delta2 = static_cast<double>(value) - mean_value;
		m2 += delta * delta2;
	}

	return count > 0.0 ? m2 / count : 0.0;
}

/** @brief Returns the population standard deviation of all elements. */
template<Array A>
double std(const A& arr)
{
	auto vars = var(arr);
	return std::sqrt(vars);
}


// Axis Reductions

/** @brief Sums values along an axis. */
template<Array A>
stratax::container::Tensor<typename A::value_type> sum(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::sum(s); });
}

/** @brief Sums values along an axis with optional dimension retention. */
template<Array A>
stratax::container::Tensor<typename A::value_type> sum(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::sum(s); }, keepdims);
}

/** @brief Multiplies values along an axis. */
template<Array A>
stratax::container::Tensor<typename A::value_type> prod(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::prod(s); });
}

/** @brief Multiplies values along an axis with optional dimension retention. */
template<Array A>
stratax::container::Tensor<typename A::value_type> prod(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::prod(s); }, keepdims);
}

/** @brief Finds maximum values along an axis. */
template<Array A>
stratax::container::Tensor<typename A::value_type> max(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::max(s); });
}

/** @brief Finds maximum values along an axis with optional dimension retention. */
template<Array A>
stratax::container::Tensor<typename A::value_type> max(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::max(s); }, keepdims);
}

/** @brief Finds minimum values along an axis. */
template<Array A>
stratax::container::Tensor<typename A::value_type> min(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::min(s); });
}

/** @brief Finds minimum values along an axis with optional dimension retention. */
template<Array A>
stratax::container::Tensor<typename A::value_type> min(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::min(s); }, keepdims);
}

/** @brief Finds flat argmax indices along an axis. */
template<Array A>
stratax::container::Tensor<std::size_t> argmax(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmax(s); });
}

/** @brief Finds flat argmax indices along an axis with optional dimension retention. */
template<Array A>
stratax::container::Tensor<std::size_t> argmax(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmax(s); }, keepdims);
}

/** @brief Finds flat argmin indices along an axis. */
template<Array A>
stratax::container::Tensor<std::size_t> argmin(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmin(s); });
}

/** @brief Finds flat argmin indices along an axis with optional dimension retention. */
template<Array A>
stratax::container::Tensor<std::size_t> argmin(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmin(s); }, keepdims);
}

/** @brief Computes means along an axis with optional dimension retention. */
template<Array A>
stratax::container::Tensor<double> mean(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::mean(s); }, keepdims);
}

/** @brief Computes means along an axis. */
template<Array A>
stratax::container::Tensor<double> mean(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::mean(s); });
}

/** @brief Computes population variances along an axis with optional dimension retention. */
template<Array A>
stratax::container::Tensor<double> var(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::var(s); }, keepdims);
}

/** @brief Computes population variances along an axis. */
template<Array A>
stratax::container::Tensor<double> var(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::var(s); });
}

/** @brief Computes population standard deviations along an axis with optional dimension retention. */
template<Array A>
stratax::container::Tensor<double> std(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::std(s); }, keepdims);
}

/** @brief Computes population standard deviations along an axis. */
template<Array A>
stratax::container::Tensor<double> std(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::std(s); });
}


}
