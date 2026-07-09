#pragma once

#include <cstddef>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>

namespace creation {

template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> zeros(const stratax::core::Shape& shape)
{
    return stratax::container::Tensor<T>(shape, T{});
}

template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> ones(const stratax::core::Shape& shape)
{
    return stratax::container::Tensor<T>(shape, T{1});
}

template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> full(const stratax::core::Shape& shape, const T& value)
{
    return stratax::container::Tensor<T>(shape, value);
}

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
