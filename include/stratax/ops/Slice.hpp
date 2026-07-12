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
#include <type_traits>
#include <utility>

namespace stratax::ops::detail {

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
    stratax::core::validation::require_at_most(
        slice.stop(),
        vec.size(),
        "Vector slice out of bounds.");

    auto begin = slice.start();
    auto end = slice.stop();

    stratax::container::Vector<T> result(slice.size());

    for (std::size_t i = begin; i < end; ++i)
    {
        result[i - begin] = vec[i];
    }

    return result;

}

template<typename T>
/**
 * @brief Copies a rectangular half-open region from a matrix.
 *
 * Rows in `[rows.start(), rows.stop())` and columns in
 * `[cols.start(), cols.stop())` are copied into a new row-major matrix.
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
    stratax::core::validation::require_at_most(
        rows.stop(),
        mat.rows(),
        "Matrix row slice out of bounds.");
    stratax::core::validation::require_at_most(
        cols.stop(),
        mat.cols(),
        "Matrix column slice out of bounds.");

    auto begin_rows = rows.start();
    auto begin_cols = cols.start();

    auto end_rows = rows.stop();
    auto end_cols = cols.stop();

    stratax::container::Matrix<T> result(rows.size(), cols.size());

    for (std::size_t i = begin_rows; i < end_rows; ++i)
    {
        for (std::size_t j = begin_cols; j < end_cols; ++j)
        {
            result(i - begin_rows, j - begin_cols) = mat(i, j);
        }
    }

    return result;
}

template<typename T, typename... Slices>
/**
 * @brief Copies a multidimensional half-open region from a tensor.
 *
 * One `Slice` must be provided for each tensor dimension. The result shape is
 * formed from the size of each slice, and values are copied in row-major order.
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

    for (std::size_t dim = 0; dim < ranges.size(); ++dim)
    {
        stratax::core::validation::require_at_most(
            ranges[dim].stop(),
            tensor.shape()(dim),
            "Tensor slice out of bounds.");
    }

    const auto result_shape = stratax::ops::detail::shape_from_slices(ranges);
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

        for (std::size_t dim = 0; dim < ranges.size(); ++dim)
        {
            const std::size_t index = remainder / result_strides(dim);
            remainder %= result_strides(dim);
            const std::size_t source_index =
                stratax::core::validation::checked_add(
                    ranges[dim].start(),
                    index,
                    "Tensor slice offset overflow.");
            const std::size_t term =
                stratax::core::validation::checked_multiply(
                    source_index,
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
