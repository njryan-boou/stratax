#pragma once

#include <cstddef>

#include <stratax/core/Shape.hpp>

namespace stratax::indexing {

/**
 * @brief Computes an unchecked flat offset from strides and normalized indices.
 *
 * Each index is multiplied by the stride for the corresponding dimension, and
 * the products are summed to produce an offset into contiguous storage.
 *
 * @tparam IndexContainer Iterable container of non-negative indices.
 * @param strides Canonical row-major strides, typically from `Shape::strides()`.
 * @param indices One normalized index for every stride.
 * @return Flat element offset represented by @p indices.
 * @pre The number of indices equals `strides.rank()`.
 * @pre Every index is valid for the shape from which @p strides was computed.
 * @pre The products and their sum are representable by `std::size_t`.
 * @complexity O(strides.rank()).
 */
template<typename IndexContainer>
std::size_t offset(
    const core::Shape& strides,
    const IndexContainer& indices)
{
    std::size_t result = 0;

    auto stride_it = strides.begin();

    for (const auto index : indices)
    {
        result += index * *stride_it;
        ++stride_it;
    }

    return result;
}

} // namespace stratax::indexing
