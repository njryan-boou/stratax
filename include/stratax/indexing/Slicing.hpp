/** @file
 * @brief Positive-step non-owning slice views with clamped bounds.
 */
// TODO: deduplicate Tensor slicing implementations.
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

/** @brief Normalized first index, signed step, and selected element count. */
struct ResolvedSlice
{
	/** @brief First normalized index; not dereferenceable for an empty slice. */
	difference_type start;
	/** @brief Nonzero signed step. */
	difference_type step;
	/** @brief Number of selected elements. */
	size_type size;
};

/** @brief Element type preserving the source array constness. */
template<typename Source>
using view_element_t = std::conditional_t<
	std::is_const_v<std::remove_reference_t<Source>>,
	const typename std::remove_cvref_t<Source>::value_type,
	typename std::remove_cvref_t<Source>::value_type>;

/**
 * @brief Normalizes explicit bounds against an extent without accessing storage.
 * @param slice Raw range; a negative-step stop of -1 is the reverse sentinel.
 * @param extent Axis length, representable by difference_type.
 * @return Clamped start, unchanged step, and selected element count.
 * @throws Exceptions::IndexError If the extent exceeds PTRDIFF_MAX.
 * @note Negative steps can be normalized, but public slice views reject them.
 * @complexity O(1).
 */
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

		const size_type count = stratax::core::Slice{start, stop, step}.size();
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

	const size_type count = stratax::core::Slice{start, stop, step}.size();
	return ResolvedSlice{start, step, count};
}

} // namespace detail

namespace detail {

/** @brief Constructs a one-axis view; see the public Vector slice overload. @internal */
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
        shape.elements() == 0 ? vec.data() : vec.data() + offset,
        shape,
        strides);
}

/** @brief Constructs a two-axis view; see the public Matrix slice overload. @internal */
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

    if (mat.strides()[0] != 0 && static_cast<size_type>(resolved_rows.step) >
        std::numeric_limits<size_type>::max() / mat.strides()[0])
    {
        throw Exceptions::DimensionError("Matrix slice stride overflow.");
    }

    const stratax::core::Shape strides{
        mat.strides()[0] * static_cast<size_type>(resolved_rows.step),
        mat.strides()[1] * static_cast<size_type>(resolved_cols.step)
    };

    return stratax::core::ArrayView<view_element_t<Matrix>>(
        shape.elements() == 0 ? mat.data() : mat.data() + offset,
        shape,
        strides);
}

} // namespace detail

/**
 * @brief Returns a non-owning Vector slice view (mutable when T is non-const).
 * @param vec Source Vector whose storage is aliased.
 * @param slice Explicit positive-step range, normalized and clamped to the vector size.
 * @return ArrayView with selected extents and source strides multiplied by each step.
 *         Every source axis is retained, including singleton and zero extents.
 * @pre The source allocation outlives the result and is not invalidated by assignment.
 * @throws Exceptions::IndexError If a step is negative or an extent exceeds PTRDIFF_MAX.
 * @throws Exceptions::DimensionError If stride/offset arithmetic or metadata validation overflows.
 * @throws std::bad_alloc If metadata allocation fails.
 * @note Bounds are clamped; a slice may be empty. Empty results retain the source data pointer.
 * @note No elements are copied. C++ views do not keep the source owner alive.
 * @complexity O(1).
 */
template<typename T>
auto slice(
	stratax::container::Vector<T>& vec,
	const stratax::core::Slice& slice)
{
	return detail::make_vector_slice_view(vec, slice);
}

/**
 * @brief Returns a non-owning Vector slice view (read-only).
 * @param vec Source Vector whose storage is aliased.
 * @param slice Explicit positive-step range, normalized and clamped to the vector size.
 * @return ArrayView with selected extents and source strides multiplied by each step.
 *         Every source axis is retained, including singleton and zero extents.
 * @pre The source allocation outlives the result and is not invalidated by assignment.
 * @throws Exceptions::IndexError If a step is negative or an extent exceeds PTRDIFF_MAX.
 * @throws Exceptions::DimensionError If stride/offset arithmetic or metadata validation overflows.
 * @throws std::bad_alloc If metadata allocation fails.
 * @note Bounds are clamped; a slice may be empty. Empty results retain the source data pointer.
 * @note No elements are copied. C++ views do not keep the source owner alive.
 * @complexity O(1).
 */
template<typename T>
auto slice(
	const stratax::container::Vector<T>& vec,
	const stratax::core::Slice& slice)
{
	return detail::make_vector_slice_view(vec, slice);
}

/**
 * @brief Returns a non-owning Matrix slice view (mutable when T is non-const).
 * @param mat Source Matrix whose storage is aliased.
 * @param rows Positive-step row range.
 * @param cols Positive-step column range.
 * @return ArrayView with selected extents and source strides multiplied by each step.
 *         Every source axis is retained, including singleton and zero extents.
 * @pre The source allocation outlives the result and is not invalidated by assignment.
 * @throws Exceptions::IndexError If a step is negative or an extent exceeds PTRDIFF_MAX.
 * @throws Exceptions::DimensionError If stride/offset arithmetic or metadata validation overflows.
 * @throws std::bad_alloc If metadata allocation fails.
 * @note Bounds are clamped; a slice may be empty. Empty results retain the source data pointer.
 * @note No elements are copied. C++ views do not keep the source owner alive.
 * @complexity O(1).
 */
template<typename T>
auto slice(
	stratax::container::Matrix<T>& mat,
	const stratax::core::Slice& rows,
	const stratax::core::Slice& cols)
{
	return detail::make_matrix_slice_view(mat, rows, cols);
}

/**
 * @brief Returns a non-owning Matrix slice view (read-only).
 * @param mat Source Matrix whose storage is aliased.
 * @param rows Positive-step row range.
 * @param cols Positive-step column range.
 * @return ArrayView with selected extents and source strides multiplied by each step.
 *         Every source axis is retained, including singleton and zero extents.
 * @pre The source allocation outlives the result and is not invalidated by assignment.
 * @throws Exceptions::IndexError If a step is negative or an extent exceeds PTRDIFF_MAX.
 * @throws Exceptions::DimensionError If stride/offset arithmetic or metadata validation overflows.
 * @throws std::bad_alloc If metadata allocation fails.
 * @note Bounds are clamped; a slice may be empty. Empty results retain the source data pointer.
 * @note No elements are copied. C++ views do not keep the source owner alive.
 * @complexity O(1).
 */
template<typename T>
auto slice(
	const stratax::container::Matrix<T>& mat,
	const stratax::core::Slice& rows,
	const stratax::core::Slice& cols)
{
	return detail::make_matrix_slice_view(mat, rows, cols);
}

namespace detail {

/** @brief Constructs a view with one slice per axis; see the public Tensor slice overloads. @internal */
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
        view_shape.elements() == 0 ? tensor.data() : tensor.data() + offset,
        view_shape,
        view_strides);
}

/** @brief Constructs a view with one slice per axis; see the public Tensor slice overloads. @internal */
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
        view_shape.elements() == 0 ? tensor.data() : tensor.data() + offset,
        view_shape,
        view_strides);
}

} // namespace detail

/**
 * @brief Returns a non-owning Tensor slice view (mutable when T is non-const).
 * @param tensor Source Tensor whose storage is aliased.
 * @param slices One explicit positive-step range per axis.
 * @return ArrayView with selected extents and source strides multiplied by each step.
 *         Every source axis is retained, including singleton and zero extents.
 * @pre The source allocation outlives the result and is not invalidated by assignment.
 * @throws Exceptions::IndexError If a step is negative or an extent exceeds PTRDIFF_MAX.
 * @throws Exceptions::RankError If the number of slices differs from the source rank.
 * @throws Exceptions::DimensionError If stride/offset arithmetic or metadata validation overflows.
 * @throws std::bad_alloc If metadata allocation fails.
 * @note Bounds are clamped; a slice may be empty. Empty results retain the source data pointer.
 * @note No elements are copied. C++ views do not keep the source owner alive.
 * @complexity O(tensor.rank()).
 */
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

/**
 * @brief Returns a non-owning Tensor slice view (read-only).
 * @param tensor Source Tensor whose storage is aliased.
 * @param slices One explicit positive-step range per axis.
 * @return ArrayView with selected extents and source strides multiplied by each step.
 *         Every source axis is retained, including singleton and zero extents.
 * @pre The source allocation outlives the result and is not invalidated by assignment.
 * @throws Exceptions::IndexError If a step is negative or an extent exceeds PTRDIFF_MAX.
 * @throws Exceptions::RankError If the number of slices differs from the source rank.
 * @throws Exceptions::DimensionError If stride/offset arithmetic or metadata validation overflows.
 * @throws std::bad_alloc If metadata allocation fails.
 * @note Bounds are clamped; a slice may be empty. Empty results retain the source data pointer.
 * @note No elements are copied. C++ views do not keep the source owner alive.
 * @complexity O(tensor.rank()).
 */
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

/**
 * @brief Returns a non-owning Tensor slice view (mutable when T is non-const).
 * @param tensor Source Tensor whose storage is aliased.
 * @param slices One explicit positive-step range per axis.
 * @return ArrayView with selected extents and source strides multiplied by each step.
 *         Every source axis is retained, including singleton and zero extents.
 * @pre The source allocation outlives the result and is not invalidated by assignment.
 * @throws Exceptions::IndexError If a step is negative or an extent exceeds PTRDIFF_MAX.
 * @throws Exceptions::RankError If the number of slices differs from the source rank.
 * @throws Exceptions::DimensionError If stride/offset arithmetic or metadata validation overflows.
 * @throws std::bad_alloc If metadata allocation fails.
 * @note Bounds are clamped; a slice may be empty. Empty results retain the source data pointer.
 * @note No elements are copied. C++ views do not keep the source owner alive.
 * @complexity O(tensor.rank()).
 */
template<typename T>
auto slice(
	stratax::container::Tensor<T>& tensor,
	const std::vector<stratax::core::Slice>& slices)
{
	return detail::make_tensor_slice_view(tensor, slices);
}

/**
 * @brief Returns a non-owning Tensor slice view (read-only).
 * @param tensor Source Tensor whose storage is aliased.
 * @param slices One explicit positive-step range per axis.
 * @return ArrayView with selected extents and source strides multiplied by each step.
 *         Every source axis is retained, including singleton and zero extents.
 * @pre The source allocation outlives the result and is not invalidated by assignment.
 * @throws Exceptions::IndexError If a step is negative or an extent exceeds PTRDIFF_MAX.
 * @throws Exceptions::RankError If the number of slices differs from the source rank.
 * @throws Exceptions::DimensionError If stride/offset arithmetic or metadata validation overflows.
 * @throws std::bad_alloc If metadata allocation fails.
 * @note Bounds are clamped; a slice may be empty. Empty results retain the source data pointer.
 * @note No elements are copied. C++ views do not keep the source owner alive.
 * @complexity O(tensor.rank()).
 */
template<typename T>
auto slice(
	const stratax::container::Tensor<T>& tensor,
	const std::vector<stratax::core::Slice>& slices)
{
	return detail::make_tensor_slice_view(tensor, slices);
}

/**
 * @brief Returns a non-owning ArrayView slice view (mutable when T is non-const).
 * @param view Source ArrayView whose storage is aliased.
 * @param slices One explicit positive-step range per axis.
 * @return ArrayView with selected extents and source strides multiplied by each step.
 *         Every source axis is retained, including singleton and zero extents.
 * @pre The source allocation outlives the result and is not invalidated by assignment.
 * @throws Exceptions::IndexError If a step is negative or an extent exceeds PTRDIFF_MAX.
 * @throws Exceptions::RankError If the number of slices differs from the source rank.
 * @throws Exceptions::DimensionError If stride/offset arithmetic or metadata validation overflows.
 * @throws std::bad_alloc If metadata allocation fails.
 * @note Bounds are clamped; a slice may be empty. Empty results retain the source data pointer.
 * @note No elements are copied. C++ views do not keep the source owner alive.
 * @complexity O(view.rank()).
 */
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
