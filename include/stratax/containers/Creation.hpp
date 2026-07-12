#pragma once

#include <cstddef>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>

namespace creation {

/**
 * @brief Creates a tensor filled with zeros.
 *
 * @tparam T Numeric value type.
 * @param shape Target shape.
 *
 * @return Tensor filled with default-constructed numeric zeros.
 */
template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> zeros(const stratax::core::Shape& shape)
{
    return stratax::container::Tensor<T>(shape, T{});
}

/**
 * @brief Creates a tensor filled with ones.
 *
 * @tparam T Numeric value type.
 * @param shape Target shape.
 *
 * @return Tensor filled with numeric ones.
 */
template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> ones(const stratax::core::Shape& shape)
{
    return stratax::container::Tensor<T>(shape, T{1});
}

/**
 * @brief Creates a tensor filled with a constant value.
 *
 * @tparam T Numeric value type.
 * @param shape Target shape.
 * @param value Value to copy into every element.
 *
 * @return Tensor filled with the requested value.
 */
template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> full(const stratax::core::Shape& shape, const T& value)
{
    return stratax::container::Tensor<T>(shape, value);
}

/**
 * @brief Creates a square identity tensor.
 *
 * @tparam T Numeric value type.
 * @param size Side length of the identity matrix.
 *
 * @return Square tensor with ones on the diagonal and zeros elsewhere.
 */
template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> identity(const std::size_t size)
{
    auto I = zeros<T>({size, size});
    for (std::size_t i = 0; i < size; ++i)
    {
        I(i, i) = T{1};
    }

    return I;
}

}
