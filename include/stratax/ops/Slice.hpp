#pragma once

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/Validation.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>
#include <type_traits>
#include <utility>

namespace stratax::ops::detail {

struct ResolvedSlice
{
    std::ptrdiff_t start;
    std::ptrdiff_t step;
    std::size_t size;
};

inline std::ptrdiff_t clamp(std::ptrdiff_t value, std::ptrdiff_t lower, std::ptrdiff_t upper)
{
    if (value < lower)
    {
        return lower;
    }
    if (value > upper)
    {
        return upper;
    }
    return value;
}

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

    if (start < 0)
    {
        start += n;
    }
    if (stop < 0)
    {
        stop += n;
    }

    if (step > 0)
    {
        start = clamp(start, 0, n);
        stop = clamp(stop, 0, n);

        if (start >= stop)
        {
            return ResolvedSlice{start, step, 0};
        }

        const std::ptrdiff_t distance = stop - start;
        const std::size_t count = static_cast<std::size_t>((distance + step - 1) / step);
        return ResolvedSlice{start, step, count};
    }

    start = clamp(start, -1, n - 1);
    stop = clamp(stop, -1, n - 1);

    if (start <= stop)
    {
        return ResolvedSlice{start, step, 0};
    }

    const std::ptrdiff_t stride = -step;
    const std::ptrdiff_t distance = start - stop;
    const std::size_t count = static_cast<std::size_t>((distance + stride - 1) / stride);
    return ResolvedSlice{start, step, count};
}

/**
 * @brief Builds a shape from a compile-time slice array.
 *
 * @tparam N Number of dimensions.
 * @tparam Is Index sequence used to expand the slice array.
 * @param ranges Slice ranges.
 *
 * @return Shape whose dimensions match the slice extents.
 */
template<std::size_t N, std::size_t... Is>
stratax::core::Shape shape_from_slices_impl(
    const std::array<stratax::core::Slice, N>& ranges,
    std::index_sequence<Is...>)
{
    return stratax::core::Shape{ranges[Is].size()...};
}

/**
 * @brief Builds a shape from a slice array.
 *
 * @tparam N Number of dimensions.
 * @param ranges Slice ranges.
 *
 * @return Shape whose dimensions match the slice extents.
 */
template<std::size_t N>
stratax::core::Shape shape_from_slices(
    const std::array<stratax::core::Slice, N>& ranges)
{
    return shape_from_slices_impl(ranges, std::make_index_sequence<N>{});
}

}

template<typename T>
/**
 * @brief Copies a half-open range from a vector.
 *
 * The returned vector contains elements in `[slice.start(), slice.stop())`
 * sampled with `slice.step()`
 * using the source vector's flat storage order.
 *
 * @param vec Source vector.
 * @param slice Half-open element range.
 *
 * @return Vector containing the selected elements.
 * @throws Exceptions::IndexError If the slice stop is greater than `vec.size()`.
 */
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

template<typename T>
/**
 * @brief Copies a rectangular half-open region from a matrix.
 *
 * Rows in `[rows.start(), rows.stop())` and columns in
 * `[cols.start(), cols.stop())` are copied into a new row-major matrix,
 * sampling each axis by its slice step.
 *
 * @param mat Source matrix.
 * @param rows Row slice.
 * @param cols Column slice.
 *
 * @return Matrix containing the selected region.
 * @throws Exceptions::IndexError If either slice exceeds the matrix bounds.
 */
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

template<typename T, typename... Slices>
/**
 * @brief Copies a multidimensional half-open region from a tensor.
 *
 * One `Slice` must be provided for each tensor dimension. The result shape is
 * formed from the size of each slice, and values are copied in row-major order
 * honoring each slice step.
 *
 * @param tensor Source tensor.
 * @param slices One slice per tensor dimension.
 *
 * @return Tensor containing the selected region.
 * @throws Exceptions::DimensionError If the number of slices does not match the tensor rank.
 * @throws Exceptions::IndexError If any slice exceeds the corresponding dimension.
 */
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
            tensor.shape()(dim),
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
            const std::size_t index = remainder / result_strides(dim);
            remainder %= result_strides(dim);

            const std::ptrdiff_t source_index =
                resolved[dim].start + static_cast<std::ptrdiff_t>(index) * resolved[dim].step;
            const std::size_t term =
                stratax::core::validation::checked_multiply(
                    static_cast<std::size_t>(source_index),
                    tensor_strides(dim),
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
