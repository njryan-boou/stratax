#pragma once

#include <pybind11/pybind11.h>

#include <stratax/core/Exceptions.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

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

inline std::size_t normalize_index(long long index, std::size_t size, const char* bounds_message)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<long long>::max()))
    {
        raise_overflow("Container size is too large to index with Python integers.");
    }

    const long long limit = static_cast<long long>(size);
    long long normalized = index;

    if (normalized < 0)
    {
        normalized += limit;
    }

    if (normalized < 0 || normalized >= limit)
    {
        throw Exceptions::IndexError(bounds_message);
    }

    return static_cast<std::size_t>(normalized);
}

inline std::size_t resolve_index(
    py::handle value,
    std::size_t size,
    const char* type_message,
    const char* overflow_message,
    const char* bounds_message)
{
    return normalize_index(cast_integer(value, type_message, overflow_message), size, bounds_message);
}

struct ResolvedSlice
{
    py::ssize_t start;
    py::ssize_t step;
    py::ssize_t length;
};

inline ResolvedSlice resolve_slice(py::slice slice, std::size_t size, const char* step_message)
{
    py::ssize_t start = 0;
    py::ssize_t stop = 0;
    py::ssize_t step = 0;
    py::ssize_t slicelength = 0;

    if (!slice.compute(static_cast<py::ssize_t>(size), &start, &stop, &step, &slicelength))
    {
        throw py::error_already_set();
    }

    if (step == 0)
    {
        throw Exceptions::TypeError(step_message);
    }

    return ResolvedSlice{start, step, slicelength};
}

inline ResolvedSlice single_index_slice(
    py::handle value,
    std::size_t size,
    const char* type_message,
    const char* overflow_message,
    const char* bounds_message)
{
    const std::size_t index = resolve_index(
        value,
        size,
        type_message,
        overflow_message,
        bounds_message);
    return ResolvedSlice{static_cast<py::ssize_t>(index), 1, 1};
}

}
