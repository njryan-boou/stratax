#pragma once

#include <cstddef>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/core/Shape.hpp>

namespace creation {

template<Numeric T>
stratax::container::Tensor<T> zeros(const stratax::core::Shape& shape)
{
    return stratax::container::Tensor<T>(shape, T{});
}

template<Numeric T>
stratax::container::Tensor<T> ones(const stratax::core::Shape& shape)
{
    return stratax::container::Tensor<T>(shape, T{1});
}

template<Numeric T>
stratax::container::Tensor<T> full(
    const stratax::core::Shape& shape,
    const T& value)
{
    return stratax::container::Tensor<T>(shape, value);
}

template<Numeric T>
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