#pragma once

#include <iterator>
#include <ostream>
#include <string>

#include <stratax/core/containers/Matrix.hpp>
#include <stratax/core/containers/Tensor.hpp>
#include <stratax/core/containers/Vector.hpp>
#include <stratax/core/algorithms/Conversions.hpp>
#include <stratax/core/containers/Shape.hpp>

namespace stratax::container {

namespace detail {

/**
 * @brief Recursively prints a tensor using nested bracket notation.
 *
 * @param os Output stream.
 * @param tensor Tensor to print.
 * @param dim Current dimension being printed.
 * @param offset Flat storage offset for the current sub-tensor.
 * @param depth Nesting depth used for indentation.
 * @param sibling_separator Separator string inserted between sibling elements.
 */
template<typename T>
void print_tensor_recursive(
    std::ostream& os,
    const Tensor<T>& tensor,
    std::size_t dim,
    std::size_t offset,
    std::size_t depth,
    const char* sibling_separator)
{
    const auto& shape = tensor.shape();
    const auto& strides = tensor.strides();

    os << "[";

    if (dim == shape.rank() - 1)
    {
        for (std::size_t i = 0; i < shape(dim); ++i)
        {
            os << tensor(offset + i * strides(dim));

            if (i + 1 != shape(dim))
                os << ", ";
        }
    }
    else
    {
        os << '\n';

        for (std::size_t i = 0; i < shape(dim); ++i)
        {
            os << std::string((depth + 1) * 4, ' ');
            print_tensor_recursive(
                os,
                tensor,
                dim + 1,
                offset + i * strides(dim),
                depth + 1,
                sibling_separator);

            if (i + 1 != shape(dim))
            {
                os << sibling_separator;
            }
        }

        os << '\n';
        os << std::string(depth * 4, ' ');
    }

    os << "]";
}

template<typename T>
std::ostream& print_tensor_like(std::ostream& os, const Tensor<T>& tensor)
{
    if (tensor.shape().elements() == 0)
    {
        os << "[]";
        return os;
    }

    print_tensor_recursive(os, tensor, 0, 0, 0, ",\n");
    return os;
}

template<typename T>
std::ostream& print_matrix_like(std::ostream& os, const Tensor<T>& tensor)
{
    print_tensor_recursive(os, tensor, 0, 0, 0, "\n");
    return os;
}

}

/**
 * @brief Writes a vector in compact bracketed list form.
 *
 * @param os Output stream.
 * @param vector Vector to print.
 *
 * @return The updated output stream.
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& vector)
{
    const auto tensor = to_tensor(vector);
    return detail::print_tensor_like(os, tensor);
}

/**
 * @brief Writes a matrix in a human-readable row-major layout.
 *
 * @param os Output stream.
 * @param matrix Matrix to print.
 *
 * @return The updated output stream.
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix)
{
    const auto tensor = to_tensor(matrix);
    return detail::print_matrix_like(os, tensor);
}

template<typename T>
/**
 * @brief Writes a tensor in nested bracket notation.
 *
 * Empty tensors are written as `[]`. Non-empty tensors are printed recursively
 * using the tensor shape and row-major strides.
 *
 * @param os Output stream.
 * @param tensor Tensor to print.
 *
 * @return The updated output stream.
 */
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor)
{
    return detail::print_tensor_like(os, tensor);
}

}
