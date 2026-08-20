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
#include <concepts>

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

template<typename Source>
using view_element_t = std::conditional_t<
	std::is_const_v<std::remove_reference_t<Source>>,
	const typename std::remove_cvref_t<Source>::value_type,
	typename std::remove_cvref_t<Source>::value_type>;

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

namespace detail {

template<typename Vector>
auto make_vector_slice_view(
    Vector& vec,
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

    return stratax::core::ArrayView<view_element_t<Vector>>(
        vec.data() + offset,
        shape,
        strides);
}

template<typename Matrix>
auto make_matrix_slice_view(
    Matrix& mat,
    const stratax::core::Slice& rows,
    const stratax::core::Slice& cols)
{
    const auto resolved_rows =
        detail::normalize_slice(
            rows,
            mat.rows(),
            "Matrix row slice out of bounds.");

    const auto resolved_cols =
        detail::normalize_slice(
            cols,
            mat.cols(),
            "Matrix column slice out of bounds.");

    if (resolved_rows.step < 0 || resolved_cols.step < 0)
    {
        throw Exceptions::IndexError(
            "Negative-step views are not supported yet.");
    }

    const size_type offset =
        static_cast<size_type>(resolved_rows.start) * mat.strides()[0]
        + static_cast<size_type>(resolved_cols.start) * mat.strides()[1];

    const stratax::core::Shape shape{
        resolved_rows.size,
        resolved_cols.size
    };

    const stratax::core::Shape strides{
        mat.strides()[0] * static_cast<size_type>(resolved_rows.step),
        mat.strides()[1] * static_cast<size_type>(resolved_cols.step)
    };

    return stratax::core::ArrayView<view_element_t<Matrix>>(
        mat.data() + offset,
        shape,
        strides);
}

} // namespace detail

template<typename T>
auto slice(
	stratax::container::Vector<T>& vec,
	const stratax::core::Slice& slice)
{
	return detail::make_vector_slice_view(vec, slice);
}

template<typename T>
auto slice(
	const stratax::container::Vector<T>& vec,
	const stratax::core::Slice& slice)
{
	return detail::make_vector_slice_view(vec, slice);
}

template<typename T>
auto slice(
	stratax::container::Matrix<T>& mat,
	const stratax::core::Slice& rows,
	const stratax::core::Slice& cols)
{
	return detail::make_matrix_slice_view(mat, rows, cols);
}

template<typename T>
auto slice(
	const stratax::container::Matrix<T>& mat,
	const stratax::core::Slice& rows,
	const stratax::core::Slice& cols)
{
	return detail::make_matrix_slice_view(mat, rows, cols);
}

namespace detail {

template<typename Tensor, typename... Slices>
requires (
    std::same_as<
        std::remove_cvref_t<Slices>,
        stratax::core::Slice
    > && ...
)
auto make_tensor_slice_view(
    Tensor& tensor,
    Slices... slices)
{
    std::array<stratax::core::Slice, sizeof...(Slices)> ranges{
        slices...
    };

    if (ranges.size() != tensor.rank())
    {
        throw Exceptions::IndexError(
            "Tensor slice rank must match tensor rank.");
    }

    std::array<
        stratax::indexing::detail::ResolvedSlice,
        sizeof...(Slices)
    > resolved{};

    std::array<size_type, sizeof...(Slices)> out_dims{};

    for (size_type dim = 0; dim < ranges.size(); ++dim)
    {
        resolved[dim] = detail::normalize_slice(
            ranges[dim],
            tensor.shape()[dim],
            "Tensor slice out of bounds.");

        out_dims[dim] = resolved[dim].size;
    }

    const auto view_shape = stratax::core::Shape(
        std::vector<size_type>(
            out_dims.begin(),
            out_dims.end()));

    const auto& tensor_strides = tensor.strides();

    size_type offset = 0;

    std::vector<size_type> view_stride_values;
    view_stride_values.reserve(resolved.size());

    for (size_type dim = 0; dim < resolved.size(); ++dim)
    {
        if (resolved[dim].step < 0)
        {
            throw Exceptions::IndexError(
                "Negative-step views are not supported yet.");
        }

        const auto start =
            static_cast<size_type>(resolved[dim].start);

        const auto offset_term =
            stratax::core::validation::checked_multiply(
                start,
                tensor_strides[dim],
                "Tensor view offset overflow.");

        offset =
            stratax::core::validation::checked_add(
                offset,
                offset_term,
                "Tensor view offset overflow.");

        view_stride_values.push_back(
            stratax::core::validation::checked_multiply(
                tensor_strides[dim],
                static_cast<size_type>(resolved[dim].step),
                "Tensor view stride overflow."));
    }

    const stratax::core::Shape view_strides(
        view_stride_values);

    return stratax::core::ArrayView<view_element_t<Tensor>>(
        tensor.data() + offset,
        view_shape,
        view_strides);
}

template<typename Tensor>
auto make_tensor_slice_view(
    Tensor& tensor,
    const std::vector<stratax::core::Slice>& slices)
{
    stratax::core::validation::require_rank(
        slices.size(),
        tensor.rank(),
        "Slice rank must match tensor rank.");

    std::vector<detail::ResolvedSlice> resolved(
        slices.size());

    std::vector<size_type> out_dims(
        slices.size());

    for (size_type dim = 0; dim < slices.size(); ++dim)
    {
        resolved[dim] = detail::normalize_slice(
            slices[dim],
            tensor.shape()[dim],
            "Tensor slice out of bounds.");

        out_dims[dim] = resolved[dim].size;
    }

    const auto view_shape =
        stratax::core::Shape(out_dims);

    const auto& tensor_strides = tensor.strides();

    size_type offset = 0;

    std::vector<size_type> view_stride_values;
    view_stride_values.reserve(resolved.size());

    for (size_type dim = 0; dim < resolved.size(); ++dim)
    {
        if (resolved[dim].step < 0)
        {
            throw Exceptions::IndexError(
                "Negative-step views are not supported yet.");
        }

        const auto start =
            static_cast<size_type>(resolved[dim].start);

        const auto offset_term =
            stratax::core::validation::checked_multiply(
                start,
                tensor_strides[dim],
                "Tensor view offset overflow.");

        offset =
            stratax::core::validation::checked_add(
                offset,
                offset_term,
                "Tensor view offset overflow.");

        view_stride_values.push_back(
            stratax::core::validation::checked_multiply(
                tensor_strides[dim],
                static_cast<size_type>(resolved[dim].step),
                "Tensor view stride overflow."));
    }

    const stratax::core::Shape view_strides(
        view_stride_values);

    return stratax::core::ArrayView<view_element_t<Tensor>>(
        tensor.data() + offset,
        view_shape,
        view_strides);
}

} // namespace detail

template<typename T, typename... Slices>
requires (
	std::same_as<
		std::remove_cvref_t<Slices>,
		stratax::core::Slice
	> && ...
)
auto slice(stratax::container::Tensor<T>& tensor, Slices... slices)
{
	return detail::make_tensor_slice_view(tensor, slices...);
}

template<typename T, typename... Slices>
requires (
	std::same_as<
		std::remove_cvref_t<Slices>,
		stratax::core::Slice
	> && ...
)
auto slice(const stratax::container::Tensor<T>& tensor, Slices... slices)
{
	return detail::make_tensor_slice_view(tensor, slices...);
}

template<typename T>
auto slice(
	stratax::container::Tensor<T>& tensor,
	const std::vector<stratax::core::Slice>& slices)
{
	return detail::make_tensor_slice_view(tensor, slices);
}

template<typename T>
auto slice(
	const stratax::container::Tensor<T>& tensor,
	const std::vector<stratax::core::Slice>& slices)
{
	return detail::make_tensor_slice_view(tensor, slices);
}

} // namespace stratax::indexing
