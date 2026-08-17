// TODO: harden offset() against rank mismatch and arithmetic overflow.
#pragma once

#include <cstddef>

#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/validation/Validation.hpp>

namespace stratax::indexing {

template<typename IndexContainer>
std::size_t offset(
    const core::Shape& shape,
    const core::Strides& strides,
    const IndexContainer& index
    )
{
    std::size_t result = 0;

    auto stride_it = strides.begin();
    auto index_it = index.begin();

    for (; index_it != index.end(); ++stride_it, ++index_it)
    {
        result += *index_it * *stride_it;
    }

    return result;
}

} // namespace stratax::indexing
