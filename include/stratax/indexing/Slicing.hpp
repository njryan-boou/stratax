#pragma once

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/validation/Validation.hpp>

#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>
#include <type_traits>
#include <utility>

namespace stratax::ops::detail {

/** @brief Normalized slice metadata for a concrete axis extent. */
struct ResolvedSlice
{
	std::ptrdiff_t start;
	std::ptrdiff_t step;
	std::size_t size;
};

inline ResolvedSlice normalize_slice(
	const stratax::core::Slice& slice,
	std::size_t extent,
	const char* message)
{
	if (extent > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
	{
		throw Exceptions::IndexError(message);
	}

	const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(extent);
	std::ptrdiff_t start = slice.start();
	std::ptrdiff_t stop = slice.stop();
	const std::ptrdiff_t step = slice.step();

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

		start = std::clamp(start, std::ptrdiff_t{0}, n);
		stop = std::clamp(stop, std::ptrdiff_t{0}, n);

		if (start >= stop)
		{
			return ResolvedSlice{start, step, 0};
		}

		const std::ptrdiff_t distance = stop - start;
		const std::size_t count = static_cast<std::size_t>((distance + step - 1) / step);
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

	start = std::clamp(start, std::ptrdiff_t{-1}, n - 1);
	stop = std::clamp(stop, std::ptrdiff_t{-1}, n - 1);

	if (start <= stop)
	{
		return ResolvedSlice{start, step, 0};
	}

	const std::ptrdiff_t stride = -step;
	const std::ptrdiff_t distance = start - stop;
	const std::size_t count = static_cast<std::size_t>((distance + stride - 1) / stride);
	return ResolvedSlice{start, step, count};
}

/** @brief Builds a shape from a compile-time slice array. */
template<std::size_t N, std::size_t... Is>
stratax::core::Shape shape_from_slices_impl(
	const std::array<stratax::core::Slice, N>& ranges,
	std::index_sequence<Is...>)
{
	return stratax::core::Shape{ranges[Is].size()...};
}

/** @brief Builds a shape from a slice array. */
template<std::size_t N>
stratax::core::Shape shape_from_slices(
	const std::array<stratax::core::Slice, N>& ranges)
{
	return shape_from_slices_impl(ranges, std::make_index_sequence<N>{});
}

}

/** @brief Copies a half-open range from a vector. */
template<typename T>
stratax::container::Vector<T>
slice(
	const stratax::container::Vector<T>& vec,
	const stratax::core::Slice& slice
)
{
	const auto resolved = stratax::ops::detail::normalize_slice(
		slice,
		vec.size(),
		"Vector slice out of bounds.");

	stratax::container::Vector<T> result(resolved.size);

	std::ptrdiff_t source = resolved.start;
	for (std::size_t i = 0; i < result.size(); ++i)
	{
		result[i] = vec[static_cast<std::size_t>(source)];
		source += resolved.step;
	}

	return result;

}

/** @brief Copies a rectangular half-open region from a matrix. */
template<typename T>
stratax::container::Matrix<T>
slice(
	const stratax::container::Matrix<T>& mat,
	const stratax::core::Slice& rows,
	const stratax::core::Slice& cols
)
{
	const auto resolved_rows = stratax::ops::detail::normalize_slice(
		rows,
		mat.rows(),
		"Matrix row slice out of bounds.");
	const auto resolved_cols = stratax::ops::detail::normalize_slice(
		cols,
		mat.cols(),
		"Matrix column slice out of bounds.");

	stratax::container::Matrix<T> result(resolved_rows.size, resolved_cols.size);

	std::ptrdiff_t source_row = resolved_rows.start;
	for (std::size_t out_row = 0; out_row < result.rows(); ++out_row)
	{
		std::ptrdiff_t source_col = resolved_cols.start;
		for (std::size_t out_col = 0; out_col < result.cols(); ++out_col)
		{
			result(out_row, out_col) = mat(
				static_cast<std::size_t>(source_row),
				static_cast<std::size_t>(source_col));
			source_col += resolved_cols.step;
		}
		source_row += resolved_rows.step;
	}

	return result;
}

/** @brief Copies a multidimensional half-open region from a tensor. */
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

	std::array<stratax::core::Slice, sizeof...(Slices)> ranges{ slices... };
    
	stratax::core::validation::require_rank(
		ranges.size(),
		tensor.rank(),
		"Slice rank must match tensor rank.");

	std::array<stratax::ops::detail::ResolvedSlice, sizeof...(Slices)> resolved{};
	std::array<std::size_t, sizeof...(Slices)> out_dims{};
	for (std::size_t dim = 0; dim < ranges.size(); ++dim)
	{
		resolved[dim] = stratax::ops::detail::normalize_slice(
			ranges[dim],
			tensor.shape()[dim],
			"Tensor slice out of bounds.");
		out_dims[dim] = resolved[dim].size;
	}

	const auto result_shape = stratax::core::Shape(
		std::vector<std::size_t>(out_dims.begin(), out_dims.end()));
	stratax::container::Tensor<T> result(result_shape);
	const stratax::core::Strides result_strides(result_shape);
	const auto& tensor_strides = tensor.strides();

	if (result.empty())
	{
		return result;
	}

	for (std::size_t flat = 0; flat < result.size(); ++flat)
	{
		std::size_t remainder = flat;
		std::size_t source_offset = 0;

		for (std::size_t dim = 0; dim < resolved.size(); ++dim)
		{
			const std::size_t index = remainder / result_strides[dim];
			remainder %= result_strides[dim];

			const std::ptrdiff_t source_index =
				resolved[dim].start + static_cast<std::ptrdiff_t>(index) * resolved[dim].step;
			const std::size_t term =
				stratax::core::validation::checked_multiply(
					static_cast<std::size_t>(source_index),
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

/** @brief Copies a multidimensional half-open region from a tensor using a vector of slices. */
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

	std::vector<stratax::ops::detail::ResolvedSlice> resolved(slices.size());
	std::vector<std::size_t> out_dims(slices.size());
    
	for (std::size_t dim = 0; dim < slices.size(); ++dim)
	{
		resolved[dim] = stratax::ops::detail::normalize_slice(
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
    
	const stratax::core::Strides result_strides(result_shape);
	const auto& tensor_strides = tensor.strides();

	for (std::size_t flat = 0; flat < result.size(); ++flat)
	{
		std::size_t remainder = flat;
		std::size_t source_offset = 0;

		for (std::size_t dim = 0; dim < resolved.size(); ++dim)
		{
			const std::size_t index = remainder / result_strides[dim];
			remainder %= result_strides[dim];

			const std::ptrdiff_t source_index =
				resolved[dim].start + static_cast<std::ptrdiff_t>(index) * resolved[dim].step;
			const std::size_t term =
				stratax::core::validation::checked_multiply(
					static_cast<std::size_t>(source_index),
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
