#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.hpp"

#include <stratax/exceptions/Exceptions.hpp>
#include "properties.hpp"
#include "arithmetic.hpp"
#include "reshape.hpp"
#include <stratax/containers/Vector.hpp>
#include <stratax/io/Print.hpp>

#include "comparison.hpp"

#include <cstddef>
#include <limits>
#include <sstream>
#include <vector>
#include <algorithm>

// =============================================================================
// Vector constructors
// =============================================================================

namespace py = pybind11;

namespace
{

using Vector = stratax::container::Vector<double>;

std::size_t checked_vector_size(long long size)
{
    if (size < 0)
    {
        throw Exceptions::DimensionError(
            "Vector size cannot be negative.");
    }

    const auto value = static_cast<std::size_t>(size);

    if (value > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        binding_utils::raise_overflow(
            "Vector storage size overflow.");
    }

    return value;
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

    Vector vector(parsed.size());
    std::copy(parsed.begin(), parsed.end(), vector.begin());

    return vector;
}

Vector make_vector_from_object(py::object value)
{
    if (py::isinstance<py::int_>(value))
    {
        return Vector(checked_vector_size(
            binding_utils::cast_integer(
                value,
                "Vector size must be an integer.",
                "Vector size is too large to fit in a signed integer.")));
    }

    if (py::isinstance<py::iterable>(value) &&
        !py::isinstance<py::str>(value))
    {
        return make_vector_from_iterable(
            value.cast<py::iterable>());
    }

    throw Exceptions::TypeError(
        "Vector constructor expects a Vector, size, or iterable of numbers.");
}

std::ptrdiff_t checked_vector_index(py::object index)
{
    return static_cast<std::ptrdiff_t>(
        binding_utils::cast_integer(
            index,
            "Vector index must be an integer.",
            "Vector index is too large to fit in a signed integer."));
}

} // anonymous namespace

void bind_vector_constructors(py::class_<Vector>& cls)
{
    cls
        .def(py::init<>())
        .def(py::init<const Vector&>(), py::arg("other"))
        .def(py::init([](py::object value) {
            return make_vector_from_object(value);
        }), py::arg("value"))
        .def(py::init([](py::object size, py::object value) {

            const auto checked_size = checked_vector_size(
                binding_utils::cast_integer(
                    size,
                    "Vector size must be an integer.",
                    "Vector size is too large to fit in a signed integer."
                )
            );

            const auto checked_value = binding_utils::cast_scalar(
                value,
                "Vector fill value must be a number.",
                "Vector fill value is too large to represent as a float."
            );

            return Vector(checked_size, checked_value);
        }),
        py::arg("size"),
        py::arg("value")
    );
}


// =============================================================================
// Vector properties
// =============================================================================

void bind_vector_properties(py::class_<Vector>& cls)
{
    bind_properties(cls);

    cls
        .def("tolist", [](const Vector& vector) {
            return std::vector<double>(vector.begin(), vector.end());
        })
        .def("__repr__", [](const Vector& vector) {
            std::ostringstream os;
            os << vector;
            return os.str();
        });
}

// =============================================================================
// Vector indexing
// =============================================================================

Vector slice_vector(const Vector& vector, const binding_utils::ResolvedSlice& slice)
{
    Vector result(static_cast<std::size_t>(slice.length));

    py::ssize_t source = slice.start;
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result[i] = vector[static_cast<std::size_t>(source)];
        source += slice.step;
    }

    return result;
}

void bind_vector_indexing(py::class_<Vector>& cls)
{
    cls
        .def("__len__", &Vector::size)
        .def("__getitem__", [](const Vector& vector, py::object index) -> py::object {
            if (py::isinstance<py::slice>(index))
            {
                const auto range = binding_utils::resolve_slice(
                    index.cast<py::slice>(),
                    vector.size(),
                    "Vector slice step cannot be zero.");
                return py::cast(slice_vector(vector, range));
            }

            return py::cast(vector.at(checked_vector_index(index)));
        })
        .def("__setitem__", [](Vector& vector, py::object index, double value) {
            vector.at(checked_vector_index(index)) = value;
        })
        .def("__iter__", [](const Vector& vector) {
            return py::make_iterator(vector.begin(), vector.end());
        }, py::keep_alive<0, 1>());
}

// =============================================================================
// Vector registration
// =============================================================================

void bind_vector(py::module_& m)
{
    py::class_<Vector> cls(m, "Vector");

    bind_vector_constructors(cls);
    bind_vector_properties(cls);
    bind_vector_indexing(cls);
    bind_arithmetic(cls);
    bind_comparison(cls);
    bind_reshape(cls);
}
