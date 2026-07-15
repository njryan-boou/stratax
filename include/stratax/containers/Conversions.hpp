#pragma once

#include <cstddef>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Validation.hpp>


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

inline stratax::core::Shape matrix_shape(const stratax::core::Shape& shape)
{
    if (shape.rank() == 2)
    {
        return shape;
    }

    std::vector<std::size_t> dims;
    dims.reserve(shape.rank());

    for (std::size_t dim : shape)
    {
        if (dim > 1)
        {
            dims.push_back(dim);
        }
    }

    return stratax::core::Shape(dims);
}

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
    if (!is_vector_shape(arr.shape()))
    {
        throw Exceptions::ShapeError(
            "Array cannot be converted to a Vector."
        );
    }

    stratax::container::Vector<typename A::value_type> result(arr.size());

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
 * @throws Exceptions::DimensionError If the source is not matrix-shaped.
 */
template<Array A>
stratax::container::Matrix<typename A::value_type>
to_matrix(const A& arr)
{
    if (!is_matrix_shape(arr.shape()))
    {
        throw Exceptions::ShapeError(
            "Array cannot be converted to a Matrix.");
    }

    const auto shape = matrix_shape(arr.shape());
    stratax::container::Matrix<typename A::value_type> result(shape);

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
