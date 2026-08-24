#pragma once

#include <cstddef>

#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::indexing {

using size_type = std::size_t;
using difference_type = std::ptrdiff_t;

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
