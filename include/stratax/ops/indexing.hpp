#pragma once

#include <cstddef>
#include <stdexcept>

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
        throw std::invalid_argument(
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
            throw std::out_of_range("Index out of bounds.");
        }

        result += (*index_it) * (*stride_it);
    }

    return result;
}
