#pragma once

#include <ostream>
#include <string>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/core/ArrayView.hpp>

namespace stratax::container {

namespace detail {

template<typename T>
void print_value(std::ostream& os, const T& value)
{
	using type = std::remove_cvref_t<T>;

	if constexpr (std::same_as<type, dtype::bool_>)
	{
		os << (value ? "true" : "false");
	}
	else if constexpr (
		std::same_as<type, dtype::int8> ||
		std::same_as<type, dtype::uint8>)
	{
		os << static_cast<int>(value);
	}
	else
	{
		os << value;
	}
}

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
	const auto logical_strides = shape.strides();

	os << "[";

	if (dim == shape.rank() - 1)
	{
		for (std::size_t i = 0; i < shape[dim]; ++i)
		{
			print_value(
	os,
	array[offset + i * logical_strides[dim]]);

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
				offset + i * logical_strides[dim],
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
std::ostream& print_array(
    std::ostream& os,
    const A& array)
{
    if (array.empty())
    {
        os << "[]";
        return os;
    }

    const char* sibling_separator =
        array.rank() == 2 ? "\n" : ",\n";

    print_recursive(
        os,
        array,
        0,
        0,
        0,
        sibling_separator);
    return os;
}

}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& vector)
{
    return detail::print_array(os, vector);
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix)
{
    return detail::print_array(os, matrix);
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor)
{
    return detail::print_array(os, tensor);
}

}

namespace stratax::core {

template<typename T>
std::ostream& operator<<(std::ostream& os, const ArrayView<T>& view)
{
    return container::detail::print_array(os, view);
}

}
