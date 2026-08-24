#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <stratax/exceptions/LayoutErrors.hpp>

namespace stratax::validation {

template<typename ShapeLike>
[[nodiscard]] Exceptions::ShapeError::shape_type shape_metadata(
    const ShapeLike& shape)
{
    return {shape.begin(), shape.end()};
}

template<typename LeftShape, typename RightShape>
inline void require_same_shape(
    const LeftShape& actual,
    const RightShape& expected)
{
    if (actual == expected) return;

    auto actual_metadata = shape_metadata(actual);
    auto expected_metadata = shape_metadata(expected);

    throw Exceptions::ShapeError(
        std::move(actual_metadata),
        std::move(expected_metadata));
}

} // namespace stratax::validation
