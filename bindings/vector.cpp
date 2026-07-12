#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/containers/Vector.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Validation.hpp>
#include <stratax/io/Print.hpp>
#include <stratax/ops/Arithmetic.hpp>
#include <stratax/ops/Comparison.hpp>

#include <cstddef>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{

using Vector = stratax::container::Vector<double>;
using Shape = stratax::core::Shape;

[[noreturn]] void raise_overflow(const char* message)
{
    PyErr_SetString(PyExc_OverflowError, message);
    throw py::error_already_set();
}

long long cast_integer(py::handle value, const char* type_message, const char* overflow_message)
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

std::size_t checked_vector_size(long long size)
{
    const std::size_t value = stratax::core::validation::nonnegative_size(
        size,
        "Vector size cannot be negative.");

    if (value > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        raise_overflow("Vector storage size overflow.");
    }

    return value;
}

double cast_scalar(py::handle value, const char* type_message, const char* overflow_message)
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

Vector make_vector_from_shape(const Shape& shape)
{
    stratax::core::validation::require_rank(shape, 1, "Vector shape must have rank 1.");
    if (shape(0) > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        raise_overflow("Vector storage size overflow.");
    }

    return Vector(shape);
}

Vector make_vector_from_iterable(py::iterable values)
{
    std::vector<double> parsed;

    for (py::handle value : values)
    {
        parsed.push_back(cast_scalar(
            value,
            "Vector values must be numbers.",
            "Vector value is too large to represent as a float."));
    }

    if (parsed.size() > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        raise_overflow("Vector storage size overflow.");
    }

    Vector vector(parsed.size());
    for (std::size_t i = 0; i < parsed.size(); ++i)
    {
        vector[i] = parsed[i];
    }

    return vector;
}

Vector make_vector_from_object(py::object value)
{
    if (py::isinstance<py::int_>(value))
    {
        return Vector(checked_vector_size(cast_integer(
            value,
            "Vector size must be an integer.",
            "Vector size is too large to fit in a signed integer.")));
    }

    if (py::isinstance<py::iterable>(value) && !py::isinstance<py::str>(value))
    {
        return make_vector_from_iterable(value.cast<py::iterable>());
    }

    throw Exceptions::TypeError(
        "Vector constructor expects a Vector, Shape, size, or iterable of numbers.");
}

} // anonymous namespace

void bind_vector(py::module_& m)
{
    py::class_<Vector>(m, "Vector")
        // Constructors
        .def(py::init([]() {
            return Vector(0);
        }))
        .def(py::init([](const Shape& shape) {
            return make_vector_from_shape(shape);
        }), py::arg("shape"))
        .def(py::init<const Vector&>(), py::arg("other"))
        .def(py::init([](py::object value) {
            return make_vector_from_object(value);
        }), py::arg("value"))
        .def(py::init([](py::object size, py::object value) {
            return Vector(checked_vector_size(cast_integer(
                size,
                "Vector size must be an integer.",
                "Vector size is too large to fit in a signed integer.")), cast_scalar(
                    value,
                    "Vector fill value must be a number.",
                    "Vector fill value is too large to represent as a float."));
        }), py::arg("size"), py::arg("value"))

        // Methods
        .def("size", &Vector::size)
        .def("rank", &Vector::rank)
        .def("empty", &Vector::empty)
        .def("shape", &Vector::shape, py::return_value_policy::reference_internal)
        .def("strides", [](const Vector& vector) {
            std::vector<std::size_t> values;
            const auto& strides = vector.strides();
            values.reserve(strides.rank());
            for (std::size_t stride : strides) {
                values.push_back(stride);
            }
            return values;
        })
        .def("fill", &Vector::fill, py::arg("value"))
        .def("tolist", [](const Vector& vector) {
            std::vector<double> values;
            values.reserve(vector.size());
            for (std::size_t i = 0; i < vector.size(); ++i) {
                values.push_back(vector[i]);
            }
            return values;
        })

        // Python special methods
        .def("__len__", &Vector::size)
        .def("__getitem__", [](const Vector& vector, py::object index) {
            return vector.at(stratax::core::validation::nonnegative_index(
                cast_integer(index, "Vector index must be an integer.", "Vector index is too large to fit in a signed integer."),
                "Vector index cannot be negative."));
        })
        .def("__setitem__", [](Vector& vector, py::object index, double value) {
            vector.at(stratax::core::validation::nonnegative_index(
                cast_integer(index, "Vector index must be an integer.", "Vector index is too large to fit in a signed integer."),
                "Vector index cannot be negative.")) = value;
        })
        .def("__iter__", [](const Vector& vector) {
            return py::make_iterator(vector.begin(), vector.end());
        }, py::keep_alive<0, 1>())
        .def("__eq__", [](const Vector& lhs, const Vector& rhs) {
            return lhs == rhs;
        })
        .def("__ne__", [](const Vector& lhs, const Vector& rhs) {
            return lhs != rhs;
        })
        .def("__add__", [](const Vector& lhs, const Vector& rhs) {
            return lhs + rhs;
        })
        .def("__add__", [](const Vector& lhs, double rhs) {
            return lhs + rhs;
        })
        .def("__radd__", [](const Vector& rhs, double lhs) {
            return lhs + rhs;
        })
        .def("__sub__", [](const Vector& lhs, const Vector& rhs) {
            return lhs - rhs;
        })
        .def("__sub__", [](const Vector& lhs, double rhs) {
            return lhs - rhs;
        })
        .def("__rsub__", [](const Vector& rhs, double lhs) {
            return lhs - rhs;
        })
        .def("__mul__", [](const Vector& lhs, const Vector& rhs) {
            return lhs * rhs;
        })
        .def("__mul__", [](const Vector& lhs, double rhs) {
            return lhs * rhs;
        })
        .def("__rmul__", [](const Vector& rhs, double lhs) {
            return lhs * rhs;
        })
        .def("__truediv__", [](const Vector& lhs, const Vector& rhs) {
            return lhs / rhs;
        })
        .def("__truediv__", [](const Vector& lhs, double rhs) {
            return lhs / rhs;
        })
        .def("__rtruediv__", [](const Vector& rhs, double lhs) {
            return lhs / rhs;
        })
        .def("__iadd__", [](Vector& lhs, const Vector& rhs) -> Vector& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__iadd__", [](Vector& lhs, double rhs) -> Vector& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Vector& lhs, const Vector& rhs) -> Vector& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Vector& lhs, double rhs) -> Vector& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Vector& lhs, const Vector& rhs) -> Vector& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Vector& lhs, double rhs) -> Vector& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Vector& lhs, const Vector& rhs) -> Vector& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Vector& lhs, double rhs) -> Vector& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__pos__", [](const Vector& vector) {
            return +vector;
        })
        .def("__neg__", [](const Vector& vector) {
            return -vector;
        })
        .def("__repr__", [](const Vector& vector) {
            std::ostringstream os;
            os << vector;
            return os.str();
        });
}
