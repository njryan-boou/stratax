// TODO: Rewrite axis_reduce to iterate directly over source strides
// instead of materializing a temporary Tensor slice for each output value.

#pragma once

#include <stratax/core/dtypes/Concepts.hpp>
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

/**
 * @brief Advances a multidimensional index in row-major order.
 * @param shape Extents that bound each component of @p indices.
 * @param indices Index updated in place, beginning with the final dimension.
 * @return `true` when another index exists; `false` after wrapping to all zeros.
 * @pre `indices.size() == shape.rank()` and every component is in range.
 * @complexity O(shape.rank()) in the worst case.
 * @internal
 */
inline bool advance(const stratax::core::Shape& shape, std::vector<std::size_t>& indices)
{
	// Start from the rightmost dimension
	for (int d = shape.rank() - 1; d >= 0; --d) {
		indices[d]++;
		if (indices[d] < shape[d]) {
			return true;  // Successfully advanced, more indices exist
		}
		indices[d] = 0;  // Reset and carry over to next dimension
	}
    
	return false;  // All dimensions wrapped around, no more indices
}

/**
 * @brief Converts a possibly negative axis to its non-negative equivalent.
 * @tparam A Stratax array type satisfying Array.
 * @param arr Array whose rank defines the negative-axis offset.
 * @param axis Axis in the range `[-arr.rank(), arr.rank() - 1]` when valid.
 * @return @p axis unchanged when non-negative, otherwise `axis + arr.rank()`.
 * @note This helper normalizes but does not validate the resulting axis.
 * @complexity O(1).
 * @internal
 */
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

/**
 * @brief Computes the dimensions produced by reducing one normalized axis.
 * @tparam A Stratax array type satisfying Array.
 * @param arr Array providing the input shape.
 * @param axis Valid, non-negative axis to reduce.
 * @param keepdims Whether to retain the reduced axis with extent one.
 * @return Input dimensions with @p axis removed or replaced by one.
 * @throws std::bad_alloc If result-dimension allocation fails.
 * @complexity O(arr.rank()).
 * @internal
 */
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

/**
 * @brief Result type produced by invoking an axis-reduction callback.
 * @tparam A Source array type satisfying Array.
 * @tparam Func Callable accepting a const Tensor of `A::value_type`.
 * @internal
 */
template<Array A, typename Func>
using axis_reduce_value_t =
	decltype(std::declval<Func>()(
		std::declval<const stratax::container::Tensor<typename A::value_type>&>()));

/**
 * @brief Applies a scalar reduction callback independently along one axis.
 *
 * Negative axes count backward from the final dimension. Each output position
 * is computed from an owning tensor slice spanning the reduced dimension. With
 * @p keepdims false the axis is removed; with it true the axis has extent one.
 * Because rank-zero tensors cannot store a scalar in the current container
 * model, reducing a rank-one array without `keepdims` returns shape `{1}`.
 * Empty output domains return immediately without invoking @p func.
 *
 * @tparam A Source Vector, Matrix, or Tensor satisfying Array.
 * @tparam Func Callable that reduces a const Tensor slice to one scalar.
 * @param array Source array.
 * @param axis Axis in `[-array.rank(), array.rank() - 1]`.
 * @param func Scalar reduction applied to every axis slice.
 * @param keepdims Whether the reduced dimension remains with extent one.
 * @return Owning tensor containing one callback result per output position.
 * @throws Exceptions::AxisError If @p axis is outside the valid range.
 * @throws Any exception propagated by conversion, slicing, allocation, or @p func.
 * @complexity O(array.size() * array.rank()) with the current slice-based implementation.
 * @internal
 */
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

	stratax::container::Tensor<typename A::value_type> arr =
		stratax::conversion::to_tensor(array);

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
	if (result.empty())
	{
		return result;
	}

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
		auto s = stratax::indexing::slice(arr, slices);
		ResultType value = func(s);
		result(output_index) = value;
	}
		while (reduction::advance(result.shape(), output_index));

	return result;
}

// Global Reductions

/**
 * @brief Returns the sum of all elements in flat iterator order.
 * @tparam A Stratax array type satisfying Array.
 * @param arr Array to reduce.
 * @return Element sum, or `A::value_type{0}` when @p arr is empty.
 * @complexity O(arr.size()).
 */
template<Array A>
typename A::value_type sum(const A& arr)
{
	return std::accumulate(
		arr.begin(),
		arr.end(),
		typename A::value_type(0)
	);
}

/**
 * @brief Returns the product of all elements in flat iterator order.
 * @tparam A Stratax array type satisfying Array.
 * @param arr Array to reduce.
 * @return Element product, or `A::value_type{1}` when @p arr is empty.
 * @complexity O(arr.size()).
 */
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

/**
 * @brief Returns the largest element.
 * @tparam A Stratax array type satisfying Array whose values are ordered.
 * @param arr Non-empty array to search.
 * @return Largest value; ties select the first occurrence.
 * @throws Exceptions::IndexError If @p arr is empty.
 * @complexity O(arr.size()).
 */
template<Array A>
typename A::value_type max(const A& arr)
{
	if (arr.empty())
	{
		throw Exceptions::IndexError(
			"Maximum is undefined for an empty array.");
	}

	auto result = std::max_element(
		arr.begin(),
		arr.end()
	);

	return *result;
}

/**
 * @brief Returns the smallest element.
 * @tparam A Stratax array type satisfying Array whose values are ordered.
 * @param arr Non-empty array to search.
 * @return Smallest value; ties select the first occurrence.
 * @throws Exceptions::IndexError If @p arr is empty.
 * @complexity O(arr.size()).
 */
template<Array A>
typename A::value_type min(const A& arr)
{
	if (arr.empty())
	{
		throw Exceptions::IndexError(
			"Minimum is undefined for an empty array.");
	}

	auto result = std::min_element(
		arr.begin(),
		arr.end()
	);

	return *result;
}

/**
 * @brief Returns the flat index of the largest element.
 * @tparam A Stratax array type satisfying Array whose values are ordered.
 * @param arr Non-empty array to search.
 * @return Zero-based row-major flat index of the first largest value.
 * @throws Exceptions::IndexError If @p arr is empty.
 * @complexity O(arr.size()).
 */
template<Array A>
std::size_t argmax(const A& arr)
{
	if (arr.empty())
	{
		throw Exceptions::IndexError(
			"Argmax is undefined for an empty array.");
	}

	auto result = std::max_element(
		arr.begin(),
		arr.end()
	);

	return static_cast<std::size_t>(std::distance(arr.begin(), result));
}

/**
 * @brief Returns the flat index of the smallest element.
 * @tparam A Stratax array type satisfying Array whose values are ordered.
 * @param arr Non-empty array to search.
 * @return Zero-based row-major flat index of the first smallest value.
 * @throws Exceptions::IndexError If @p arr is empty.
 * @complexity O(arr.size()).
 */
template<Array A>
std::size_t argmin(const A& arr)
{
	if (arr.empty())
	{
		throw Exceptions::IndexError(
			"Argmin is undefined for an empty array.");
	}

	auto result = std::min_element(
		arr.begin(),
		arr.end()
	);

	return static_cast<std::size_t>(std::distance(arr.begin(), result));
}

/**
 * @brief Returns the arithmetic mean of all elements as double.
 * @tparam A Stratax array type satisfying Array and convertible to double.
 * @param arr Non-empty array to reduce.
 * @return `sum(arr) / arr.size()` converted to double.
 * @throws Exceptions::ZeroDivisionError If @p arr is empty.
 * @complexity O(arr.size()).
 */
template<Array A>
double mean(const A& arr)
{
	if (arr.empty())
	{
		throw Exceptions::ZeroDivisionError(
			"Mean is undefined for an empty array.");
	}

	return static_cast<double>(sum(arr)) / static_cast<double>(arr.size());
}

/**
 * @brief Returns the population variance using Welford's online algorithm.
 * @tparam A Stratax array type satisfying Array and convertible to double.
 * @param arr Non-empty array to reduce.
 * @return Sum of squared deviations divided by `arr.size()`.
 * @throws Exceptions::ZeroDivisionError If @p arr is empty.
 * @complexity O(arr.size()).
 */
template<Array A>
double var(const A& arr)
{
	if (arr.empty())
	{
		throw Exceptions::ZeroDivisionError(
			"Variance is undefined for an empty array.");
	}

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

	return m2 / count;
}

/**
 * @brief Returns the population standard deviation of all elements.
 * @tparam A Stratax array type satisfying Array and convertible to double.
 * @param arr Non-empty array to reduce.
 * @return Square root of `var(arr)`.
 * @throws Exceptions::ZeroDivisionError If @p arr is empty.
 * @complexity O(arr.size()).
 */
template<Array A>
double std(const A& arr)
{
	auto vars = var(arr);
	return std::sqrt(vars);
}


// Axis Reductions

/**
 * @brief Sums values along an axis and removes that dimension.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @return Tensor shaped as @p arr without @p axis; empty slices produce zero.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<typename A::value_type> sum(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::sum(s); });
}

/**
 * @brief Sums values along an axis with optional dimension retention.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @param keepdims Retains the reduced axis with extent one when true.
 * @return Tensor of per-slice sums; empty slices produce zero.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<typename A::value_type> sum(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::sum(s); }, keepdims);
}

/**
 * @brief Multiplies values along an axis and removes that dimension.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @return Tensor of per-slice products; empty slices produce one.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<typename A::value_type> prod(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::prod(s); });
}

/**
 * @brief Multiplies values along an axis with optional dimension retention.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @param keepdims Retains the reduced axis with extent one when true.
 * @return Tensor of per-slice products; empty slices produce one.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<typename A::value_type> prod(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::prod(s); }, keepdims);
}

/**
 * @brief Finds maximum values along an axis and removes that dimension.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @return Tensor containing the first maximum value from each slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::IndexError If a produced output requires reducing an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<typename A::value_type> max(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::max(s); });
}

/**
 * @brief Finds maximum values along an axis with optional dimension retention.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @param keepdims Retains the reduced axis with extent one when true.
 * @return Tensor containing the first maximum value from each slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::IndexError If a produced output requires reducing an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<typename A::value_type> max(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::max(s); }, keepdims);
}

/**
 * @brief Finds minimum values along an axis and removes that dimension.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @return Tensor containing the first minimum value from each slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::IndexError If a produced output requires reducing an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<typename A::value_type> min(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::min(s); });
}

/**
 * @brief Finds minimum values along an axis with optional dimension retention.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @param keepdims Retains the reduced axis with extent one when true.
 * @return Tensor containing the first minimum value from each slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::IndexError If a produced output requires reducing an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<typename A::value_type> min(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::min(s); }, keepdims);
}

/**
 * @brief Finds indices of maximum values along an axis.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @return Tensor containing the first maximum's zero-based index within each axis slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::IndexError If a produced output requires reducing an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<std::size_t> argmax(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmax(s); });
}

/**
 * @brief Finds indices of maximum values with optional dimension retention.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @param keepdims Retains the reduced axis with extent one when true.
 * @return Tensor containing the first maximum's zero-based index within each axis slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::IndexError If a produced output requires reducing an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<std::size_t> argmax(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmax(s); }, keepdims);
}

/**
 * @brief Finds indices of minimum values along an axis.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @return Tensor containing the first minimum's zero-based index within each axis slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::IndexError If a produced output requires reducing an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<std::size_t> argmin(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmin(s); });
}

/**
 * @brief Finds indices of minimum values with optional dimension retention.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @param keepdims Retains the reduced axis with extent one when true.
 * @return Tensor containing the first minimum's zero-based index within each axis slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::IndexError If a produced output requires reducing an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<std::size_t> argmin(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmin(s); }, keepdims);
}

/**
 * @brief Computes arithmetic means along an axis with optional dimension retention.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @param keepdims Retains the reduced axis with extent one when true.
 * @return Tensor<double> containing one mean per axis slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::ZeroDivisionError If a produced output has an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<double> mean(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::mean(s); }, keepdims);
}

/**
 * @brief Computes arithmetic means along an axis and removes that dimension.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @return Tensor<double> containing one mean per axis slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::ZeroDivisionError If a produced output has an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<double> mean(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::mean(s); });
}

/**
 * @brief Computes population variances with optional dimension retention.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @param keepdims Retains the reduced axis with extent one when true.
 * @return Tensor<double> containing one population variance per slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::ZeroDivisionError If a produced output has an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<double> var(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::var(s); }, keepdims);
}

/**
 * @brief Computes population variances along an axis and removes that dimension.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @return Tensor<double> containing one population variance per slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::ZeroDivisionError If a produced output has an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<double> var(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::var(s); });
}

/**
 * @brief Computes population standard deviations with optional dimension retention.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @param keepdims Retains the reduced axis with extent one when true.
 * @return Tensor<double> containing one population standard deviation per slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::ZeroDivisionError If a produced output has an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<double> std(const A& arr, int axis, bool keepdims)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::std(s); }, keepdims);
}

/**
 * @brief Computes population standard deviations and removes the reduced dimension.
 * @param arr Source array. @param axis Axis to reduce; negative values count from the end.
 * @return Tensor<double> containing one population standard deviation per slice.
 * @throws Exceptions::AxisError If @p axis is invalid.
 * @throws Exceptions::ZeroDivisionError If a produced output has an empty slice.
 * @complexity O(arr.size() * arr.rank()).
 */
template<Array A>
stratax::container::Tensor<double> std(const A& arr, int axis)
{
	return axis_reduce(arr, axis, [](const auto& s) { return reduction::std(s); });
} // namespace reduction


}
