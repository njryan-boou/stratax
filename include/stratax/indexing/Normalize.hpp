#pragma once

#include <cstddef>

#include <stratax/core/validation/IndexValidation.hpp>

namespace stratax::indexing {

using size_type = std::size_t;
using difference_type = std::ptrdiff_t;

inline size_type normalize_index(
    difference_type index,
    size_type size)
{
    return validation::normalize_index(index, size);
}

} // namespace stratax::indexing
