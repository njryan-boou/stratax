#pragma once

#include <cstddef>
#include <limits>

#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/Shape.hpp>

template<typename IndexContainer>
std::size_t offset(
    const stratax::core::Shape& shape,
    const stratax::core::Strides& strides,
    const IndexContainer& index)
{
    if (shape.rank() != strides.rank() || shape.rank() != index.size())
    {
        throw Exceptions::DimensionError(
            "Shape, strides, and index must have the same rank.");
    }

    std::size_t result = 0;

    auto shape_it = shape.begin();
    auto stride_it = strides.begin();
    auto index_it = index.begin();

    for (; index_it != index.end(); ++shape_it, ++stride_it, ++index_it)
    {
        if (*index_it >= *shape_it)
        {
            throw Exceptions::IndexError("Index out of bounds.");
        }

        if (*stride_it != 0 &&
            *index_it > std::numeric_limits<std::size_t>::max() / *stride_it)
        {
            throw Exceptions::IndexError("Index offset overflow.");
        }

        const std::size_t term = (*index_it) * (*stride_it);

        if (result > std::numeric_limits<std::size_t>::max() - term)
        {
            throw Exceptions::IndexError("Index offset overflow.");
        }

        result += term;
    }

    return result;
}
