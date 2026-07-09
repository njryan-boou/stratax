#pragma once

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Config.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Vector.hpp>

template<Array A>
stratax::container::Tensor<typename A::value_type>
reshape(const A& arr, const stratax::core::Shape& shape)
{
    if (arr.size() != shape.elements())
    {
        throw Exceptions::ShapeError("Shape mismatch.");
    }

    stratax::container::Tensor<typename A::value_type> result(shape);

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result[i] = arr[i];
    }

    return result;
}

template<Array A>
stratax::container::Vector<typename A::value_type> 
flatten(const A& arr)
{
    stratax::container::Vector<typename A::value_type> result(arr.size());

    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        result[i] = arr[i];
    }

    return result;
}
