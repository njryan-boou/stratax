
#pragma once

#include <iterator>
#include <ostream>
#include <string>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/Shape.hpp>

namespace stratax::container {

template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& vector)
{
    os << "[";

    const auto end = vector.end();
    for (auto it = vector.begin(); it != end; ++it)
    {
        os << *it;

        if (std::next(it) != end)
        {
            os << ", ";
        }
    }

    os << "]";

    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix)
{
    os << "[\n";

    for (std::size_t i = 0; i < matrix.rows(); ++i)
    {
        os << "    [";

        for (std::size_t j = 0; j < matrix.cols(); ++j)
        {
            
            os << matrix(i, j);

            if (j + 1 != matrix.cols())
                os << ", ";
        }

        os << "]";

        if (i + 1 != matrix.rows())
            os << '\n';
    }

    os << '\n' << "]";

    return os;
}

template<typename T>
void print_recursive(
    std::ostream& os,
    const Tensor<T>& tensor,
    std::size_t dim,
    std::size_t offset,
    std::size_t depth
)
{
    const auto& shape = tensor.shape();
    const auto& strides = tensor.strides();

    os << "[";

    // Base case: last dimension
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
            print_recursive(
                os,
                tensor,
                dim + 1,
                offset + i * strides(dim),
                depth + 1
                );

            if (i + 1 != shape(dim))
            {
                os << ",\n";
            }
        }
        os << '\n';
        os << std::string(depth * 4, ' ');
    }
    os << "]";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor)
{
    if (tensor.shape().elements() == 0)
    {
        os << "[]";
        return os;
    }

    print_recursive(os, tensor, 0, 0, 0);

    return os;
}

} 
