#pragma once

#include <pybind11/pybind11.h>

#include <stratax/exceptions/ArithmeticErrors.hpp>
#include <stratax/exceptions/TypeErrors.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/indexing/Indexing.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace binding_utils {

namespace py = pybind11;

[[noreturn]] inline void raise_overflow(
    const Exceptions::OverflowError& error)
{
    PyErr_SetString(PyExc_OverflowError, error.what());
    throw py::error_already_set();
}

inline long long cast_integer(py::handle value)
{
    if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value))
    {
        throw Exceptions::TypeError::expected_integer();
    }

    PyErr_Clear();
    const long long result = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred())
    {
        PyErr_Clear();
        raise_overflow(Exceptions::OverflowError::integer());
    }

    return result;
}

inline std::ptrdiff_t cast_index(py::handle value)
{
    const long long result = cast_integer(value);

    if constexpr (sizeof(std::ptrdiff_t) < sizeof(long long))
    {
        if (result < std::numeric_limits<std::ptrdiff_t>::min() ||
            result > std::numeric_limits<std::ptrdiff_t>::max())
        {
            raise_overflow(Exceptions::OverflowError::integer());
        }
    }

    return static_cast<std::ptrdiff_t>(result);
}

inline double cast_scalar(py::handle value)
{
    if (py::isinstance<py::bool_>(value)
        || !(py::isinstance<py::int_>(value) || py::isinstance<py::float_>(value)))
    {
        throw Exceptions::TypeError::expected_number();
    }

    PyErr_Clear();
    const double result = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred())
    {
        PyErr_Clear();
        raise_overflow(Exceptions::OverflowError::floating());
    }

    if (!std::isfinite(result))
    {
        raise_overflow(Exceptions::OverflowError::floating());
    }

    return result;
}

inline stratax::core::Slice single_index_slice(
    py::handle value,
    std::size_t size)
{
    const std::ptrdiff_t index = cast_index(value);

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
            Exceptions::OverflowError::python_slice_extent());
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
