#pragma once

#include <ostream>
#include <string>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

namespace stratax::container {

namespace detail {

/** @brief Recursively prints a tensor using nested bracket notation. */
template<Array A>
void print_recursive(
    std::ostream& os,
    const A& array,
    std::size_t dim,
    std::size_t offset,
    std::size_t depth,
    const char* sibling_separator)
{
	const auto& shape = array.shape();
	const auto& strides = array.strides();

	os << "[";

	if (dim == shape.rank() - 1)
	{
		for (std::size_t i = 0; i < shape[dim]; ++i)
		{
			os << array[offset + i * strides[dim]];

			if (i + 1 != shape[dim])
				os << ", ";
		}
	}
	else
	{
		os << '\n';

		for (std::size_t i = 0; i < shape[dim]; ++i)
		{
			os << std::string((depth + 1) * 4, ' ');
			print_recursive(
				os,
				array,
				dim + 1,
				offset + i * strides[dim],
				depth + 1,
				sibling_separator);

			if (i + 1 != shape[dim])
			{
				os << sibling_separator;
			}
		}

		os << '\n';
		os << std::string(depth * 4, ' ');
	}

	os << "]";
}

template<Array A>
std::ostream& print_tensor_like(
    std::ostream& os,
    const A& array)
{
    if (array.empty())
    {
        os << "[]";
        return os;
    }

    print_recursive(os, array, 0, 0, 0, ",\n");
    return os;
}

template<Array A>
std::ostream& print_matrix_like(
    std::ostream& os,
    const A& array)
{
    if (array.empty())
    {
        os << "[]";
        return os;
    }

    print_recursive(os, array, 0, 0, 0, "\n");
    return os;
}

}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& vector)
{
    return detail::print_tensor_like(os, vector);
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix)
{
    return detail::print_matrix_like(os, matrix);
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor)
{
    return detail::print_tensor_like(os, tensor);
}

}
