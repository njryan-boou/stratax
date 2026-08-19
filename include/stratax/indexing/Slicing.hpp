// TODO: deduplicate Tensor slicing implementations.
// TODO: make slice normalization arithmetic overflow-safe.
// TODO: improve Tensor slice error messages.
// TODO: support omitted slice bounds for NumPy-style slicing.
// TODO: revisit signed strides when implementing views.
#pragma once

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/core/ArrayView.hpp>

#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>
#include <type_traits>
#include <utility>

namespace stratax::indexing {

/** @brief Unsigned type used for extents, counts, and flat offsets. */
using size_type = std::size_t;
/** @brief Signed type used for normalized slice positions and steps. */
using difference_type = std::ptrdiff_t;

namespace detail
{

struct ResolvedSlice
{
	difference_type start;
	difference_type step;
	size_type size;
};

inline ResolvedSlice normalize_slice(
	const stratax::core::Slice& slice,
	size_type extent,
	const char* message)
{
	if (extent > static_cast<size_type>(std::numeric_limits<difference_type>::max()))
	{
		throw Exceptions::IndexError(message);
	}

	const difference_type n = static_cast<difference_type>(extent);
	difference_type start = slice.start();
	difference_type stop = slice.stop();
	const difference_type step = slice.step();

	if (step > 0)
	{
		if (start < 0)
		{
			start += n;
		}
		if (stop < 0)
		{
			stop += n;
		}

		start = std::clamp(start, difference_type{0}, n);
		stop = std::clamp(stop, difference_type{0}, n);

		if (start >= stop)
		{
			return ResolvedSlice{start, step, 0};
		}

		const difference_type distance = stop - start;
		const size_type count = static_cast<size_type>((distance + step - 1) / step);
		return ResolvedSlice{start, step, count};
	}

	if (start < 0)
	{
		start += n;
	}
	if (stop < 0 && stop != -1)
	{
		stop += n;
	}

	start = std::clamp(start, difference_type{-1}, n - 1);
	stop = std::clamp(stop, difference_type{-1}, n - 1);

	if (start <= stop)
	{
		return ResolvedSlice{start, step, 0};
	}

	const difference_type stride = -step;
	const difference_type distance = start - stop;
	const size_type count = static_cast<size_type>((distance + stride - 1) / stride);
	return ResolvedSlice{start, step, count};
}

} // namespace detail

template<typename T>
stratax::core::ArrayView<T>
slice(
    stratax::container::Vector<T>& vec,
    const stratax::core::Slice& slice)
{
    const auto resolved =
        detail::normalize_slice(
            slice,
            vec.size(),
            "Vector slice out of bounds.");

    if (resolved.step < 0)
    {
        throw Exceptions::IndexError(
            "Negative-step views are not supported yet.");
    }

    const auto offset =
        static_cast<size_type>(resolved.start);

    const stratax::core::Shape shape{
        resolved.size
    };

    const stratax::core::Shape strides{
        static_cast<size_type>(resolved.step)
    };

    return stratax::core::ArrayView<T>(
        vec.data() + offset,
        shape,
        strides);
}

template<typename T>
stratax::container::Matrix<T>
slice(
	const stratax::container::Matrix<T>& mat,
	const stratax::core::Slice& rows,
	const stratax::core::Slice& cols
)
{
	const auto resolved_rows = detail::normalize_slice(
		rows,
		mat.rows(),
		"Matrix row slice out of bounds.");
	const auto resolved_cols = detail::normalize_slice(
		cols,
		mat.cols(),
		"Matrix column slice out of bounds.");

	stratax::container::Matrix<T> result(resolved_rows.size, resolved_cols.size);

	difference_type source_row = resolved_rows.start;
	for (size_type out_row = 0; out_row < result.rows(); ++out_row)
	{
		difference_type source_col = resolved_cols.start;
		for (size_type out_col = 0; out_col < result.cols(); ++out_col)
		{
			result(out_row, out_col) = mat(
				static_cast<size_type>(source_row),
				static_cast<size_type>(source_col));
			source_col += resolved_cols.step;
		}
		source_row += resolved_rows.step;
	}

	return result;
}

template<typename T, typename... Slices>
stratax::container::Tensor<T>
slice(
	const stratax::container::Tensor<T>& tensor,
	Slices... slices
)
{
	static_assert(
		(std::is_same_v<Slices, stratax::core::Slice> && ...),
		"All arguments must be Slice."
	);

	std::array<stratax::core::Slice, sizeof...(Slices)> ranges{slices...};

	if (ranges.size() != tensor.rank())
	{
		throw Exceptions::IndexError(
			"Tensor slice rank must match tensor rank.");
	}

	std::array<stratax::indexing::detail::ResolvedSlice, sizeof...(Slices)> resolved{};
	std::array<size_type, sizeof...(Slices)> out_dims{};
	for (size_type dim = 0; dim < ranges.size(); ++dim)
	{
		resolved[dim] = detail::normalize_slice(
			ranges[dim],
			tensor.shape()[dim],
			"Tensor slice out of bounds.");
		out_dims[dim] = resolved[dim].size;
	}

	const auto result_shape = stratax::core::Shape(
		std::vector<size_type>(out_dims.begin(), out_dims.end()));
	stratax::container::Tensor<T> result(result_shape);
	const stratax::core::Shape result_strides = result_shape.strides();
	const auto& tensor_strides = tensor.strides();

	if (result.empty())
	{
		return result;
	}

	for (size_type flat = 0; flat < result.size(); ++flat)
	{
		size_type remainder = flat;
		size_type source_offset = 0;

		for (size_type dim = 0; dim < resolved.size(); ++dim)
		{
			const size_type index = remainder / result_strides[dim];
			remainder %= result_strides[dim];

			const difference_type source_index =
				resolved[dim].start + static_cast<difference_type>(index) * resolved[dim].step;
			const size_type term =
				stratax::core::validation::checked_multiply(
					static_cast<size_type>(source_index),
					tensor_strides[dim],
					"Tensor slice offset overflow.");
			source_offset =
				stratax::core::validation::checked_add(
					source_offset,
					term,
					"Tensor slice offset overflow.");
		}

		result[flat] = tensor[source_offset];
	}

	return result;
}

template<typename T>
stratax::container::Tensor<T>
slice(
	const stratax::container::Tensor<T>& tensor,
	const std::vector<stratax::core::Slice>& slices
)
{
	stratax::core::validation::require_rank(
		slices.size(),
		tensor.rank(),
		"Slice rank must match tensor rank.");

	std::vector<detail::ResolvedSlice> resolved(slices.size());
	std::vector<size_type> out_dims(slices.size());

	for (size_type dim = 0; dim < slices.size(); ++dim)
	{
		resolved[dim] = detail::normalize_slice(
			slices[dim],
			tensor.shape()[dim],
			"Tensor slice out of bounds.");
		out_dims[dim] = resolved[dim].size;
	}

	const auto result_shape = stratax::core::Shape(out_dims);
	stratax::container::Tensor<T> result(result_shape);

	if (result.empty())
	{
		return result;
	}

	const stratax::core::Shape result_strides = result_shape.strides();
	const auto& tensor_strides = tensor.strides();

	for (size_type flat = 0; flat < result.size(); ++flat)
	{
		size_type remainder = flat;
		size_type source_offset = 0;

		for (size_type dim = 0; dim < resolved.size(); ++dim)
		{
			const size_type index = remainder / result_strides[dim];
			remainder %= result_strides[dim];

			const difference_type source_index =
				resolved[dim].start + static_cast<difference_type>(index) * resolved[dim].step;
			const size_type term =
				stratax::core::validation::checked_multiply(
					static_cast<size_type>(source_index),
					tensor_strides[dim],
					"Tensor slice offset overflow.");
			source_offset =
				stratax::core::validation::checked_add(
					source_offset,
					term,
					"Tensor slice offset overflow.");
		}

		result[flat] = tensor[source_offset];
	}

	return result;
}

} // namespace stratax::indexing
