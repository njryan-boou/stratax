#pragma once

#include <cstddef>

#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Shape.hpp>

namespace stratax::creation {

/**
 * @brief Creates a tensor filled with additive identity values.
 * @tparam T Numeric element type.
 * @param shape Shape of the resulting tensor.
 * @return Owning tensor with @p shape whose elements equal `T{}`.
 * @throws Exceptions::DimensionError If element-count or stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(shape.elements() + shape.rank()).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Tensor<T> zeros(const stratax::core::Shape& shape)
{
    return stratax::container::Tensor<T>(shape, T{});
}

/**
 * @brief Creates a vector filled with additive identity values.
 * @tparam T Numeric element type.
 * @param size Number of elements in the resulting vector.
 * @return Owning rank-one vector containing @p size copies of `T{}`.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(size).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Vector<T> zeros(std::size_t size)
{
    return stratax::container::Vector<T>(size, T{});
}

/**
 * @brief Creates a matrix filled with additive identity values.
 * @tparam T Numeric element type.
 * @param rows Number of result rows.
 * @param cols Number of result columns.
 * @return Owning matrix of shape `{rows, cols}` filled with `T{}`.
 * @throws Exceptions::DimensionError If element-count or stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(rows * cols).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Matrix<T> zeros(std::size_t rows, std::size_t cols)
{
    return stratax::container::Matrix<T>(rows, cols, T{});
}

/**
 * @brief Creates a tensor filled with multiplicative identity values.
 * @tparam T Numeric element type.
 * @param shape Shape of the resulting tensor.
 * @return Owning tensor with @p shape whose elements equal `T{1}`.
 * @throws Exceptions::DimensionError If element-count or stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(shape.elements() + shape.rank()).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Tensor<T> ones(const stratax::core::Shape& shape)
{
    return stratax::container::Tensor<T>(shape, T{1});
}

/**
 * @brief Creates a vector filled with multiplicative identity values.
 * @tparam T Numeric element type.
 * @param size Number of elements in the resulting vector.
 * @return Owning rank-one vector containing @p size copies of `T{1}`.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(size).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Vector<T> ones(std::size_t size)
{
    return stratax::container::Vector<T>(size, T{1});
}

/**
 * @brief Creates a matrix filled with multiplicative identity values.
 * @tparam T Numeric element type.
 * @param rows Number of result rows.
 * @param cols Number of result columns.
 * @return Owning matrix of shape `{rows, cols}` filled with `T{1}`.
 * @throws Exceptions::DimensionError If element-count or stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(rows * cols).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Matrix<T> ones(std::size_t rows, std::size_t cols)
{
    return stratax::container::Matrix<T>(rows, cols, T{1});
}

/**
 * @brief Creates a tensor filled with a specified value.
 * @tparam T Numeric element type.
 * @param shape Shape of the resulting tensor.
 * @param value Value copied into every element.
 * @return Owning tensor with @p shape filled with @p value.
 * @throws Exceptions::DimensionError If element-count or stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(shape.elements() + shape.rank()).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Tensor<T> full(const stratax::core::Shape& shape, const T& value)
{
    return stratax::container::Tensor<T>(shape, value);
}

/**
 * @brief Creates a vector filled with a specified value.
 * @tparam T Numeric element type.
 * @param size Number of elements in the resulting vector.
 * @param value Value copied into every element.
 * @return Owning rank-one vector containing @p size copies of @p value.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(size).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Vector<T> full(std::size_t size, const T& value)
{
	return stratax::container::Vector<T>(size, value);
}

/**
 * @brief Creates a matrix filled with a specified value.
 * @tparam T Numeric element type.
 * @param rows Number of result rows.
 * @param cols Number of result columns.
 * @param value Value copied into every element.
 * @return Owning matrix of shape `{rows, cols}` filled with @p value.
 * @throws Exceptions::DimensionError If element-count or stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(rows * cols).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Matrix<T> full(std::size_t rows, std::size_t cols, const T& value)
{
    return stratax::container::Matrix<T>(rows, cols, value);
}

/**
 * @brief Creates a square identity matrix.
 *
 * Off-diagonal elements are initialized to `T{}` and each diagonal element is
 * assigned `T{1}`.
 *
 * @tparam T Numeric element type.
 * @param size Number of rows and columns.
 * @return Owning matrix of shape `{size, size}` containing the identity.
 * @throws Exceptions::DimensionError If `size * size` or stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(size * size).
 */
template<Numeric T>
[[nodiscard]]
stratax::container::Matrix<T> identity(std::size_t size)
{
    stratax::container::Matrix<T> result(size, size, T{});

    for (std::size_t i = 0; i < size; ++i)
    {
        result(i, i) = T{1};
    }

    return result;
}

}
