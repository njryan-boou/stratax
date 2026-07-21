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



namespace reduction {

// Global Reductions

template<Array A>
typename A::value_type sum(const A& arr)
{
    return std::accumulate(
        arr.begin(),
        arr.end(),
        0
    );
}

template<Array A>
typename A::value_type prod(const A& arr)
{
    return std::accumulate(
        arr.begin(),
        arr.end(),
        1,
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


// Axis Reductions


stratax::container::Vector<int> sum(const stratax::container::Matrix<int>& arr, int axis)
{
    if (axis == 0)
    {
         stratax::container::Matrix<int> result(1, arr.cols());
        for (std::size_t i = 0; i < arr.cols(); i ++)
        {
            result(0, i) = sum(slice(arr, Slice{0, arr.rows()}, Slice{0 + i, 1 + i}));
        }
    }
}
}