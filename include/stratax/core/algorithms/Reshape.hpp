#pragma once

#include <stratax/core/containers/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/containers/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/core/containers/Tensor.hpp>
#include <stratax/core/containers/Matrix.hpp>
#include <stratax/core/containers/Vector.hpp>

/**
 * @brief Reshapes an array-like object into a tensor with the requested shape.
 *
 * The element order is preserved while the logical shape changes.
 *
 * @param arr Source array.
 * @param shape Target shape.
 *
 * @return Tensor containing the same values in the new shape.
 *
 * @throws Exceptions::ShapeError If the shape does not match the number of elements.
 */
template<Array A>
stratax::container::Tensor<typename A::value_type>
reshape(const A& arr, const stratax::core::Shape& shape)
{
    stratax::core::validation::require_equal_size(
        arr.size(),
        shape.elements(),
        "Shape mismatch.");

    stratax::container::Tensor<typename A::value_type> result(shape);

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result[i] = arr[i];
    }

    return result;
}

/**
 * @brief Flattens an array-like object into a vector.
 *
 * The values are copied in storage order.
 *
 * @param arr Source array.
 *
 * @return Vector containing the flattened values.
 */
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

