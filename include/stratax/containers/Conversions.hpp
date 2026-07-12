#pragma once

#include <cstddef>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Validation.hpp>

/**
 * @brief Converts an array-like object to a vector.
 *
 * @tparam A Source array type.
 * @param arr Source array.
 *
 * @return Vector containing the same values and shape.
 *
 * @throws Exceptions::DimensionError If the source is not rank 1.
 */
template<Array A>
stratax::container::Vector<typename A::value_type>
to_vector(const A& arr)
{
    stratax::core::validation::require_rank(
        arr,
        1,
        "Cannot convert non-rank-1 array to Vector");

    stratax::container::Vector<typename A::value_type> result(arr.shape());

    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        result[i] = arr[i];
    }

    return result;
}

/**
 * @brief Converts an array-like object to a matrix.
 *
 * @tparam A Source array type.
 * @param arr Source array.
 *
 * @return Matrix containing the same values and shape.
 *
 * @throws Exceptions::DimensionError If the source is not rank 2.
 */
template<Array A>
stratax::container::Matrix<typename A::value_type>
to_matrix(const A& arr)
{
    stratax::core::validation::require_rank(
        arr,
        2,
        "Cannot convert non-rank-2 array to Matrix");

    stratax::container::Matrix<typename A::value_type> result(arr.shape());

    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        result[i] = arr[i];
    }

    return result;
}

/**
 * @brief Converts an array-like object to a tensor.
 *
 * @tparam A Source array type.
 * @param arr Source array.
 *
 * @return Tensor containing the same values and shape.
 */
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

/**
 * @brief Casts a vector to a different numeric value type.
 *
 * @tparam To Target numeric type.
 * @tparam From Source numeric type.
 * @param vec Source vector.
 *
 * @return Vector with each element cast to the target type.
 */
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

/**
 * @brief Casts a matrix to a different numeric value type.
 *
 * @tparam To Target numeric type.
 * @tparam From Source numeric type.
 * @param mat Source matrix.
 *
 * @return Matrix with each element cast to the target type.
 */
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

/**
 * @brief Casts a tensor to a different numeric value type.
 *
 * @tparam To Target numeric type.
 * @tparam From Source numeric type.
 * @param tensor Source tensor.
 *
 * @return Tensor with each element cast to the target type.
 */
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
