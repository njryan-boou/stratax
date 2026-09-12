/** @file
 * @brief Checked signed index normalization.
 */
#pragma once

#include <cstddef>

#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::indexing {

/** @brief Unsigned extent and normalized-index type. */
using size_type = std::size_t;
/** @brief Signed type for Python-style indices. */
using difference_type = std::ptrdiff_t;

/**
 * @brief Validates a signed index and converts it to a non-negative offset.
 * @param index Index in `[-size, size)`; negative values count from the end.
 * @param size Number of elements in the indexed dimension.
 * @return The non-negative index; `-1` selects `size - 1` when nonempty.
 * @throws Exceptions::IndexError If the index is out of bounds, including every index when size is zero.
 * @note The minimum representable signed index is handled without signed overflow.
 * @complexity O(1).
 */
inline size_type normalize_index(
    difference_type index,
    size_type size)
{
	const size_type magnitude = index < 0
		? static_cast<size_type>(-(index + 1)) + 1
		: 0;

	if ((index >= 0 && static_cast<size_type>(index) >= size) ||
		(index < 0 && magnitude > size))
	{
		throw Exceptions::IndexError("Index is out of bounds.");
	}

	return index >= 0 ? static_cast<size_type>(index) : size - magnitude;
}

} // namespace stratax::indexing
