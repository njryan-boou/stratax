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
	size_type extent)
{
	if (extent > static_cast<size_type>(std::numeric_limits<difference_type>::max()))
	{
		throw Exceptions::IndexError("Slice extent is too large.");
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
            vec.size());

    if (resolved.step < 0)
    {
        throw Exceptions::IndexError(
            "Negative-step views are not supported.");
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
            mat.rows());

    const auto resolved_cols =
        detail::normalize_slice(
            cols,
            mat.cols());

    if (resolved_rows.step < 0 || resolved_cols.step < 0)
    {
        throw Exceptions::IndexError(
            "Negative-step views are not supported.");
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
        throw Exceptions::RankError(
            "The number of slices must match the tensor rank.");
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
            tensor.shape()[dim]);

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
                "Negative-step views are not supported.");
        }

        const auto start =
            static_cast<size_type>(resolved[dim].start);

        if (tensor_strides[dim] != 0 &&
            start > std::numeric_limits<size_type>::max() / tensor_strides[dim])
        {
            throw Exceptions::DimensionError("Tensor slice offset overflow.");
        }

        const auto offset_term = start * tensor_strides[dim];

        if (offset > std::numeric_limits<size_type>::max() - offset_term)
        {
            throw Exceptions::DimensionError("Tensor slice offset overflow.");
        }

        offset += offset_term;

        const auto step = static_cast<size_type>(resolved[dim].step);

        if (step != 0 &&
            tensor_strides[dim] > std::numeric_limits<size_type>::max() / step)
        {
            throw Exceptions::DimensionError("Tensor slice stride overflow.");
        }

        view_stride_values.push_back(tensor_strides[dim] * step);
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
    if (slices.size() != tensor.rank())
    {
        throw Exceptions::RankError(
            "The number of slices must match the tensor rank.");
    }

    std::vector<detail::ResolvedSlice> resolved(
        slices.size());

    std::vector<size_type> out_dims(
        slices.size());

    for (size_type dim = 0; dim < slices.size(); ++dim)
    {
        resolved[dim] = detail::normalize_slice(
            slices[dim],
            tensor.shape()[dim]);

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
                "Negative-step views are not supported.");
        }

        const auto start =
            static_cast<size_type>(resolved[dim].start);

        if (tensor_strides[dim] != 0 &&
            start > std::numeric_limits<size_type>::max() / tensor_strides[dim])
        {
            throw Exceptions::DimensionError("Tensor slice offset overflow.");
        }

        const auto offset_term = start * tensor_strides[dim];

        if (offset > std::numeric_limits<size_type>::max() - offset_term)
        {
            throw Exceptions::DimensionError("Tensor slice offset overflow.");
        }

        offset += offset_term;

        const auto step = static_cast<size_type>(resolved[dim].step);

        if (step != 0 &&
            tensor_strides[dim] > std::numeric_limits<size_type>::max() / step)
        {
            throw Exceptions::DimensionError("Tensor slice stride overflow.");
        }

        view_stride_values.push_back(tensor_strides[dim] * step);
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

template<typename T>
stratax::core::ArrayView<T>
slice(
    stratax::core::ArrayView<T>& view,
    const std::vector<stratax::core::Slice>& slices)
{
    if (slices.size() != view.rank())
    {
        throw Exceptions::RankError(
            "The number of slices must match the view rank.");
    }

    std::vector<detail::ResolvedSlice> resolved(slices.size());
    std::vector<size_type> out_dims(slices.size());
    std::vector<size_type> out_strides(slices.size());

    size_type start_offset = 0;

    for (size_type dim = 0; dim < slices.size(); ++dim)
    {
        resolved[dim] = detail::normalize_slice(
            slices[dim],
            view.shape()[dim]);

        out_dims[dim] = resolved[dim].size;

        if (resolved[dim].step < 0)
        {
            throw Exceptions::IndexError(
                "Negative-step views are not supported.");
        }

        const size_type start =
            static_cast<size_type>(resolved[dim].start);

        const size_type step =
            static_cast<size_type>(resolved[dim].step);

        if (view.strides()[dim] != 0 &&
            start >
                std::numeric_limits<size_type>::max() /
                view.strides()[dim])
        {
            throw Exceptions::DimensionError(
                "ArrayView slice offset overflow.");
        }

        const size_type start_term =
            start * view.strides()[dim];

        if (start_offset >
            std::numeric_limits<size_type>::max() -
                start_term)
        {
            throw Exceptions::DimensionError(
                "ArrayView slice offset overflow.");
        }

        start_offset += start_term;

        if (step != 0 &&
            view.strides()[dim] >
                std::numeric_limits<size_type>::max() /
                step)
        {
            throw Exceptions::DimensionError(
                "ArrayView slice stride overflow.");
        }

        out_strides[dim] =
            view.strides()[dim] * step;
    }

    const stratax::core::Shape out_shape(out_dims);
    const stratax::core::Shape out_stride_shape(out_strides);

    auto* data = view.data();

    if (out_shape.elements() != 0)
    {
        data += start_offset;
    }

    return stratax::core::ArrayView<T>(
        data,
        out_shape,
        out_stride_shape);
}

} // namespace stratax::indexing
