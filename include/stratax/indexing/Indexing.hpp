#pragma once

#include <cstddef>

#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>

namespace stratax::indexing {

/** @brief Computes a flat storage offset from a shape, strides, and index container. */
template<typename IndexContainer>
std::size_t offset(
	const stratax::core::Shape& shape,
	const stratax::core::Strides& strides,
	const IndexContainer& index)
{
	stratax::core::validation::require_rank(
		strides.rank(),
		shape.rank(),
		"Indexing requires shape, strides, and index to have the same rank.");
	stratax::core::validation::require_rank(
		index.size(),
		shape.rank(),
		"Indexing requires shape, strides, and index to have the same rank.");

	std::size_t result = 0;

	auto shape_it = shape.begin();
	auto stride_it = strides.begin();
	auto index_it = index.begin();

	for (; index_it != index.end(); ++shape_it, ++stride_it, ++index_it)
	{
		if (*index_it >= *shape_it)
		{
			throw Exceptions::IndexError("Index component is out of bounds.");
		}

		std::size_t term = 0;

		try
		{
			term = stratax::core::validation::checked_multiply(
				*index_it,
				*stride_it,
				"Index offset multiplication overflow.");
		}
		catch (const Exceptions::DimensionError&)
		{
			throw Exceptions::IndexError("Index offset multiplication overflow.");
		}

		try
		{
			result = stratax::core::validation::checked_add(
				result,
				term,
				"Index offset addition overflow.");
		}
		catch (const Exceptions::DimensionError&)
		{
			throw Exceptions::IndexError("Index offset addition overflow.");
		}
	}

	return result;
}

inline std::size_t normalize_index(
    std::ptrdiff_t index,
    std::size_t size)
{
    if (index >= 0)
    {
        const auto normalized = static_cast<std::size_t>(index);

        if (normalized >= size)
        {
            throw Exceptions::IndexError("Index is out of bounds.");
        }

        return normalized;
    }

    const std::size_t magnitude =
        static_cast<std::size_t>(-(index + 1)) + 1;

    if (magnitude > size)
    {
        throw Exceptions::IndexError("Index is out of bounds.");
    }

    return size - magnitude;
}

} // namespace stratax::indexing
