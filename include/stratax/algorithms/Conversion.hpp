#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/ArrayTraits.hpp>
#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::conversion {

namespace detail {

/**
 * @brief Determines whether a shape can be represented by a Vector.
 *
 * Every rank-one shape is accepted. Higher-rank shapes are accepted only when
 * exactly one dimension is greater than one. Dimensions of size zero or one do
 * not count as non-singleton dimensions.
 *
 * @param shape Shape to inspect.
 * @return `true` when @p shape is vector-compatible; otherwise `false`.
 * @complexity O(shape.rank()).
 * @internal
 */
inline bool is_vector_shape(const stratax::core::Shape& shape)
{
    if (shape.rank() == 1)
    {
        return true;
    }

    std::size_t non_singleton = 0;

    for (std::size_t dim : shape)
    {
        if (dim > 1)
        {
            ++non_singleton;
        }
    }

    return non_singleton == 1;
}

/**
 * @brief Determines whether a shape can be represented by a Matrix.
 *
 * Every rank-two shape is accepted. Shapes of any other rank are accepted only
 * when exactly two dimensions are greater than one. Dimensions of size zero or
 * one do not count as non-singleton dimensions.
 *
 * @param shape Shape to inspect.
 * @return `true` when @p shape is matrix-compatible; otherwise `false`.
 * @complexity O(shape.rank()).
 * @internal
 */
inline bool is_matrix_shape(const stratax::core::Shape& shape)
{
    if (shape.rank() == 2)
    {
        return true;
    }

    std::size_t non_singleton = 0;

    for (std::size_t dim : shape)
    {
        if (dim > 1)
        {
            ++non_singleton;
        }
    }

    return non_singleton == 2;
}

/**
 * @brief Produces the rank-two shape used by a matrix conversion.
 *
 * Rank-two shapes are returned unchanged. For other ranks, dimensions greater
 * than one are retained in their original order while zero and singleton
 * dimensions are removed. Callers must first establish matrix compatibility
 * with is_matrix_shape().
 *
 * @param shape Matrix-compatible source shape.
 * @return Two-dimensional shape used by the resulting Matrix.
 * @throws std::bad_alloc If storage for the filtered dimensions cannot be allocated.
 * @complexity O(shape.rank()).
 * @internal
 */
inline stratax::core::Shape matrix_shape(const stratax::core::Shape& shape)
{
    if (shape.rank() == 2)
    {
        return shape;
    }

    std::vector<std::size_t> dims;
    dims.reserve(2);

    for (std::size_t dim : shape)
    {
        if (dim > 1)
        {
            dims.push_back(dim);
        }
    }

    return stratax::core::Shape(dims);
}

} // namespace detail

/**
 * @brief Copies a vector-compatible Stratax array into a Vector.
 *
 * The result contains the source elements in flat row-major iterator order and
 * owns storage independent of @p arr. Rank-one sources are always compatible;
 * other ranks must contain exactly one dimension greater than one.
 *
 * @tparam A Stratax array type satisfying Array.
 * @param arr Source array to convert.
 * @return Owning Vector with the same value type and element count as @p arr.
 * @throws Exceptions::ShapeError If the source shape is not vector-compatible.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(arr.size() + arr.rank()).
 */
template<Array A>
[[nodiscard]]
stratax::container::Vector<typename A::value_type>
to_vector(const A& arr)
{
    if (!detail::is_vector_shape(arr.shape()))
    {
        throw Exceptions::ShapeError(
            "Array cannot be converted to a Vector.");
    }

    stratax::container::Vector<typename A::value_type> result(arr.size());

    std::copy(
        arr.begin(),
        arr.end(),
        result.begin());

    return result;
}

/**
 * @brief Copies a matrix-compatible Stratax array into a Matrix.
 *
 * Rank-two sources retain their shape. For other ranks, zero and singleton
 * dimensions are removed and the two remaining non-singleton dimensions form
 * the result shape. Values retain flat row-major iterator order, and the result
 * owns independent storage.
 *
 * @tparam A Stratax array type satisfying Array.
 * @param arr Source array to convert.
 * @return Owning Matrix with the same value type as @p arr.
 * @throws Exceptions::ShapeError If the source shape is not matrix-compatible.
 * @throws Exceptions::DimensionError If result shape arithmetic overflows.
 * @throws std::bad_alloc If result or shape allocation fails.
 * @complexity O(arr.size() + arr.rank()).
 */
template<Array A>
[[nodiscard]]
stratax::container::Matrix<typename A::value_type>
to_matrix(const A& arr)
{
    if (!detail::is_matrix_shape(arr.shape()))
    {
        throw Exceptions::ShapeError(
            "Array cannot be converted to a Matrix.");
    }

    const stratax::core::Shape shape =
        detail::matrix_shape(arr.shape());

    stratax::container::Matrix<typename A::value_type> result(shape);

    std::copy(
        arr.begin(),
        arr.end(),
        result.begin());

    return result;
}

/**
 * @brief Copies a Stratax array into a Tensor of the same shape.
 *
 * Values retain flat row-major iterator order. The returned tensor preserves
 * the source shape exactly and owns storage independent of @p arr.
 *
 * @tparam A Stratax array type satisfying Array.
 * @param arr Source array to convert.
 * @return Owning Tensor with the same value type, shape, and values as @p arr.
 * @throws Exceptions::DimensionError If result stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(arr.size() + arr.rank()).
 */
template<Array A>
[[nodiscard]]
stratax::container::Tensor<typename A::value_type>
to_tensor(const A& arr)
{
    stratax::container::Tensor<typename A::value_type> result(arr.shape());

    std::copy(
        arr.begin(),
        arr.end(),
        result.begin());

    return result;
}

/**
 * @brief Casts every element of a Stratax array to another dtype.
 *
 * The result preserves the source container category and shape while replacing
 * its element type with @p To. Each element is converted with
 * `static_cast<To>`, so the standard C++ rules for narrowing, truncation,
 * signedness conversion, boolean conversion, and precision loss apply.
 *
 * The returned array owns storage independent of @p arr.
 *
 * @tparam To Destination Stratax dtype.
 * @tparam A Source Stratax array type satisfying Array.
 * @param arr Array whose elements are converted.
 * @return Owning array with the same container category and shape as @p arr,
 *         with element type @p To.
 * @throws Exceptions::DimensionError If result shape or stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(arr.size() + arr.rank()).
 */
template<DType To, Array A>
[[nodiscard]]
stratax::core::rebind_array_t<A, To>
astype(const A& arr)
{
    using result_type =
        stratax::core::rebind_array_t<A, To>;

    result_type result(arr.shape());

    std::transform(
        arr.begin(),
        arr.end(),
        result.begin(),
        [](const auto& value)
        {
            return static_cast<To>(value);
        });

    return result;
}

} // namespace stratax::conversion