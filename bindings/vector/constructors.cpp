#include <pybind11/pybind11.h>

#include "../common/utils.hpp"

#include <stratax/containers/Vector.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Validation.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace
{

using Vector = stratax::container::Vector<double>;
using Shape = stratax::core::Shape;

std::size_t checked_vector_size(long long size)
{
    const std::size_t value = stratax::core::validation::nonnegative_size(
        size,
        "Vector size cannot be negative.");

    if (value > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        binding_utils::raise_overflow("Vector storage size overflow.");
    }

    return value;
}

Vector make_vector_from_shape(const Shape& shape)
{
    stratax::core::validation::require_rank(shape, 1, "Vector shape must have rank 1.");
    if (shape(0) > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        binding_utils::raise_overflow("Vector storage size overflow.");
    }

    return Vector(shape);
}

Vector make_vector_from_iterable(py::iterable values)
{
    std::vector<double> parsed;

    for (py::handle value : values)
    {
        parsed.push_back(binding_utils::cast_scalar(
            value,
            "Vector values must be numbers.",
            "Vector value is too large to represent as a float."));
    }

    if (parsed.size() > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        binding_utils::raise_overflow("Vector storage size overflow.");
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
        return Vector(checked_vector_size(binding_utils::cast_integer(
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

void bind_vector_constructors(py::class_<Vector>& cls)
{
    cls
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
            return Vector(checked_vector_size(binding_utils::cast_integer(
                size,
                "Vector size must be an integer.",
                "Vector size is too large to fit in a signed integer.")), binding_utils::cast_scalar(
                    value,
                    "Vector fill value must be a number.",
                    "Vector fill value is too large to represent as a float."));
        }), py::arg("size"), py::arg("value"));
}
