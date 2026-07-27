#pragma once

#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/containers/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/core/containers/Tensor.hpp>
#include <stratax/core/containers/Matrix.hpp>
#include <stratax/core/containers/Vector.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/core/ops/Slice.hpp>

#include <numeric>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

namespace reduction {

// Advance multi-dimensional indices in row-major order
// Advances indices to the next valid combination within the shape
// Returns true if more indices exist, false if we've gone past the end
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

template<Array A, typename Func>
auto axis_reduce(const A& array, int axis, Func func)
{
    using ResultType =
    decltype(func(std::declval<const stratax::container::Tensor<typename A::value_type>&>()));

    if (axis < 0 || axis >= static_cast<int>(array.rank()))
    {
        throw Exceptions::AxisError("axis is out of range.");
    }

    stratax::container::Tensor<typename A::value_type> arr = to_tensor(array);

    stratax::core::Shape input_shape = arr.shape();

    std::vector<std::size_t> result_dimensions;

    for (std::size_t dimension = 0; dimension < input_shape.rank(); ++dimension)
    {
        if (dimension != static_cast<std::size_t>(axis))
        {
            result_dimensions.push_back(input_shape[dimension]);
        }
    }

    // Handle rank-0 result (reduction to scalar) - wrap in rank-0 tensor
    if (result_dimensions.empty())
    {
        stratax::container::Tensor<typename A::value_type> full_arr = arr;
        ResultType scalar_result = func(full_arr);
        stratax::container::Tensor<ResultType> result(stratax::core::Shape{});
        result[0] = scalar_result;  // Flat indexing for rank-0 tensor
        return result;
    }

    stratax::container::Tensor<ResultType> result(stratax::core::Shape{result_dimensions});

    std::vector<std::size_t> output_index(result.rank(), 0);
    std::vector<stratax::core::Slice> slices;
    
    do {
        slices.clear();
        std::size_t output_position = 0;
        for (std::size_t dimension = 0; dimension < input_shape.rank(); ++dimension)
        {
        
            if (static_cast<int>(dimension) == axis)
            {
                slices.push_back(stratax::core::Slice{static_cast<std::ptrdiff_t>(0), 
                static_cast<std::ptrdiff_t>(input_shape[dimension])});
            }

            else
            {
                slices.push_back(stratax::core::Slice{static_cast<std::ptrdiff_t>(output_index[output_position]), 
                static_cast<std::ptrdiff_t>(output_index[output_position] + 1)});
                output_position++;
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

template<Array A>
typename A::value_type sum(const A& arr)
{
    return std::accumulate(
        arr.begin(),
        arr.end(),
        typename A::value_type(0)
    );
}

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

template<Array A>
typename A::value_type max(const A& arr)
{
    auto result = std::max_element(
        arr.begin(),
        arr.end()
    );

    return *result;
}

template<Array A>
typename A::value_type min(const A& arr)
{
    auto result = std::min_element(
        arr.begin(),
        arr.end()
    );

    return *result;
}

template<Array A>
std::size_t argmax(const A& arr)
{
    auto result = std::max_element(
        arr.begin(),
        arr.end()
    );

    return static_cast<std::size_t>(std::distance(arr.begin(), result));
}

template<Array A>
std::size_t argmin(const A& arr)
{
    auto result = std::min_element(
        arr.begin(),
        arr.end()
    );

    return static_cast<std::size_t>(std::distance(arr.begin(), result));
}

template<Array A>
double mean(const A& arr)
{
    return static_cast<double>(sum(arr)) / static_cast<double>(arr.size());
}

template<Array A>
double var(const A& arr)
{
    auto diff = arr - mean(arr);
    auto diff_sqr = std::pow(diff, static_cast<double>(2));

    return diff_sqr / static_cast<double>(arr.size());
}

template<Array A>
double std(const A& arr)
{
    auto vars = var(arr);
    return std::sqrt(vars);
}


// Axis Reductions

template<Array A>
stratax::container::Tensor<typename A::value_type> sum(const A& arr, int axis)
{
    return axis_reduce(arr, axis, [](const auto& s) { return reduction::sum(s); });
}

template<Array A>
stratax::container::Tensor<typename A::value_type> prod(const A& arr, int axis)
{
    return axis_reduce(arr, axis, [](const auto& s) { return reduction::prod(s); });
}

template<Array A>
stratax::container::Tensor<typename A::value_type> max(const A& arr, int axis)
{
    return axis_reduce(arr, axis, [](const auto& s) { return reduction::max(s); });
}

template<Array A>
stratax::container::Tensor<typename A::value_type> min(const A& arr, int axis)
{
    return axis_reduce(arr, axis, [](const auto& s) { return reduction::min(s); });
}

template<Array A>
stratax::container::Tensor<std::size_t> argmax(const A& arr, int axis)
{
    return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmax(s); });
}

template<Array A>
stratax::container::Tensor<std::size_t> argmin(const A& arr, int axis)
{
    return axis_reduce(arr, axis, [](const auto& s) { return reduction::argmin(s); });
}

template<Array A>
stratax::container::Tensor<double> mean(const A& arr, int axis)
{
    return axis_reduce(arr, axis, [](const auto& s) { return reduction::mean(s); });
}

template<Array A>
stratax::container::Tensor<double> var(const A& arr, int axis)
{
    return axis_reduce(arr, axis, [](const auto& s) { return reduction::var(s); });
}

template<Array A>
stratax::container::Tensor<double> std(const A& arr, int axis)
{
    return axis_reduce(arr, axis, [](const auto& s) { return reduction::std(s); });
}


}