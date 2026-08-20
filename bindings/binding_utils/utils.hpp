#pragma once

#include <pybind11/pybind11.h>

#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/indexing/Indexing.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace binding_utils {

namespace py = pybind11;

[[noreturn]] inline void raise_overflow(const char* message)
{
    PyErr_SetString(PyExc_OverflowError, message);
    throw py::error_already_set();
}

inline long long cast_integer(py::handle value, const char* type_message, const char* overflow_message)
{
    if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value))
    {
        throw Exceptions::TypeError(type_message);
    }

    PyErr_Clear();
    const long long result = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred())
    {
        PyErr_Clear();
        raise_overflow(overflow_message);
    }

    return result;
}

inline std::ptrdiff_t cast_index(
    py::handle value,
    const char* type_message,
    const char* overflow_message)
{
    const long long result =
        cast_integer(value, type_message, overflow_message);

    if constexpr (sizeof(std::ptrdiff_t) < sizeof(long long))
    {
        if (result < std::numeric_limits<std::ptrdiff_t>::min() ||
            result > std::numeric_limits<std::ptrdiff_t>::max())
        {
            raise_overflow(overflow_message);
        }
    }

    return static_cast<std::ptrdiff_t>(result);
}

inline double cast_scalar(py::handle value, const char* type_message, const char* overflow_message)
{
    if (py::isinstance<py::bool_>(value)
        || !(py::isinstance<py::int_>(value) || py::isinstance<py::float_>(value)))
    {
        throw Exceptions::TypeError(type_message);
    }

    PyErr_Clear();
    const double result = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred())
    {
        PyErr_Clear();
        raise_overflow(overflow_message);
    }

    if (!std::isfinite(result))
    {
        raise_overflow(overflow_message);
    }

    return result;
}

inline stratax::core::Slice single_index_slice(
    py::handle value,
    std::size_t size,
    const char* type_message,
    const char* overflow_message)
{
    const std::ptrdiff_t index = cast_index(
        value,
        type_message,
        overflow_message);

    const std::size_t normalized =
        stratax::indexing::normalize_index(index, size);

    return stratax::core::Slice(
        static_cast<std::ptrdiff_t>(normalized),
        static_cast<std::ptrdiff_t>(normalized + 1));
}

inline stratax::core::Slice cast_slice(
    py::slice slice,
    std::size_t size)
{
    if (size > static_cast<std::size_t>(
            std::numeric_limits<py::ssize_t>::max()))
    {
        raise_overflow(
            "Container is too large to slice with Python indices.");
    }

    py::ssize_t start;
    py::ssize_t stop;
    py::ssize_t step;
    py::ssize_t length;

    if (!slice.compute(
            static_cast<py::ssize_t>(size),
            &start,
            &stop,
            &step,
            &length))
    {
        throw py::error_already_set();
    }

    return stratax::core::Slice(
        static_cast<std::ptrdiff_t>(start),
        static_cast<std::ptrdiff_t>(stop),
        static_cast<std::ptrdiff_t>(step));
}

template<typename Tensor>
py::object tensor_to_list_recursive(
    const Tensor& tensor,
    std::size_t dimension,
    std::size_t offset)
{
    py::list values;
    const auto& shape = tensor.shape();

    for (std::size_t i = 0; i < shape[dimension]; ++i)
    {
        const std::size_t index =
            offset + i * tensor.strides()[dimension];

        if (dimension + 1 == shape.rank())
        {
            values.append(tensor[index]);
        }
        else
        {
            values.append(tensor_to_list_recursive(
                tensor,
                dimension + 1,
                index));
        }
    }

    return std::move(values);
}

template<typename Tensor>
py::list tensor_to_list(const Tensor& tensor)
{
    if (tensor.rank() == 0)
    {
        return py::list();
    }

    return tensor_to_list_recursive(tensor, 0, 0).template cast<py::list>();
}

} // namespace binding_utils
