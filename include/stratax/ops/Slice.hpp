#pragma once

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/core/Strides.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace stratax::ops::detail {

template<std::size_t N, std::size_t... Is>
stratax::core::Shape shape_from_slices_impl(
    const std::array<stratax::core::Slice, N>& ranges,
    std::index_sequence<Is...>)
{
    return stratax::core::Shape{ranges[Is].size()...};
}

template<std::size_t N>
stratax::core::Shape shape_from_slices(
    const std::array<stratax::core::Slice, N>& ranges)
{
    return shape_from_slices_impl(ranges, std::make_index_sequence<N>{});
}

}

template<typename T>
stratax::container::Vector<T>
slice(
    const stratax::container::Vector<T>& vec,
    const stratax::core::Slice& slice
)
{
    if (slice.stop() > vec.size())
    {
        throw Exceptions::IndexError("Vector slice out of bounds.");
    }

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
stratax::container::Matrix<T>
slice(
    const stratax::container::Matrix<T>& mat,
    const stratax::core::Slice& rows,
    const stratax::core::Slice& cols
)
{
    if (rows.stop() > mat.rows())
    {
        throw Exceptions::IndexError("Matrix row slice out of bounds.");
    }

    if (cols.stop() > mat.cols())
    {
        throw Exceptions::IndexError("Matrix column slice out of bounds.");
    }

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
    
    if (ranges.size() != tensor.rank())
    {
        throw Exceptions::DimensionError("Slice rank must match tensor rank.");
    }

    for (std::size_t dim = 0; dim < ranges.size(); ++dim)
    {
        if (ranges[dim].stop() > tensor.shape()(dim))
        {
            throw Exceptions::IndexError("Tensor slice out of bounds.");
        }
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
            source_offset += (ranges[dim].start() + index) * tensor_strides(dim);
        }

        result[flat] = tensor[source_offset];
    }

    return result;
}
