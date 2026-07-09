#pragma once

#include <cstddef>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>

template<Array A>
stratax::container::Vector<typename A::value_type>
to_vector(const A& arr)
{
    if (arr.rank() != 1)
    {
        throw Exceptions::DimensionError("Cannot convert non-rank-1 array to Vector");
    }

    stratax::container::Vector<typename A::value_type> result(arr.shape());

    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        result[i] = arr[i];
    }

    return result;
}

template<Array A>
stratax::container::Matrix<typename A::value_type>
to_matrix(const A& arr)
{
    if (arr.rank() != 2)
    {
        throw Exceptions::DimensionError("Cannot convert non-rank-2 array to Matrix");
    }

    stratax::container::Matrix<typename A::value_type> result(arr.shape());

    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        result[i] = arr[i];
    }

    return result;
}

template<Array A>
stratax::container::Tensor<typename A::value_type>
to_tensor(const A& arr)
{
    stratax::container::Tensor<typename A::value_type> result(arr.shape());

    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        result[i] = arr[i];
    }

    return result;
}

template<typename To, typename From>
requires Numeric<To> && Numeric<From>
stratax::container::Vector<To>
astype(const stratax::container::Vector<From>& vec)
{
    stratax::container::Vector<To> result(vec.shape());

    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        result[i] = static_cast<To>(vec[i]);
    }

    return result;
}

template<typename To, typename From>
requires Numeric<To> && Numeric<From>
stratax::container::Matrix<To>
astype(const stratax::container::Matrix<From>& mat)
{
    stratax::container::Matrix<To> result(mat.shape());

    for (std::size_t i = 0; i < mat.size(); ++i)
    {
        result[i] = static_cast<To>(mat[i]);
    }

    return result;
}

template<typename To, typename From>
requires Numeric<To> && Numeric<From>
stratax::container::Tensor<To>
astype(const stratax::container::Tensor<From>& tensor)
{
    stratax::container::Tensor<To> result(tensor.shape());

    for (std::size_t i = 0; i < tensor.size(); ++i)
    {
        result[i] = static_cast<To>(tensor[i]);
    }

    return result;
}
