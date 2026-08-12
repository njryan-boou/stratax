#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.hpp"

#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/algorithms/Reshape.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/io/Print.hpp>
#include <stratax/ops/Arithmetic.hpp>
#include <stratax/ops/Comparison.hpp>

#include <cstddef>
#include <sstream>
#include <vector>

// =============================================================================
// Vector constructors
// =============================================================================

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


// =============================================================================
// Vector properties
// =============================================================================


namespace
{

using Vector = stratax::container::Vector<double>;

} // anonymous namespace

void bind_vector_properties(py::class_<Vector>& cls)
{
    cls
        .def_property_readonly("size", &Vector::size)
        .def_property_readonly("rank", &Vector::rank)
        .def_property_readonly("empty", &Vector::empty)
        .def_property_readonly("shape", &Vector::shape, py::return_value_policy::reference_internal)
        .def_property_readonly("strides", [](const Vector& vector) {
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
        .def("__repr__", [](const Vector& vector) {
            std::ostringstream os;
            os << vector;
            return os.str();
        });
}

// =============================================================================
// Vector indexing
// =============================================================================


namespace
{

using Vector = stratax::container::Vector<double>;

Vector slice_vector_runtime(const Vector& vector, const binding_utils::ResolvedSlice& slice)
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

} // anonymous namespace

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
                return py::cast(slice_vector_runtime(vector, range));
            }

            return py::cast(vector.at(binding_utils::resolve_index(
                index,
                vector.size(),
                "Vector index must be an integer.",
                "Vector index is too large to fit in a signed integer.",
                "Vector index is out of bounds.")));
        })
        .def("__setitem__", [](Vector& vector, py::object index, double value) {
            vector.at(binding_utils::resolve_index(
                index,
                vector.size(),
                "Vector index must be an integer.",
                "Vector index is too large to fit in a signed integer.",
                "Vector index is out of bounds.")) = value;
        })
        .def("__iter__", [](const Vector& vector) {
            return py::make_iterator(vector.begin(), vector.end());
        }, py::keep_alive<0, 1>());
}

// =============================================================================
// Vector arithmetic
// =============================================================================


namespace
{

using Vector = stratax::container::Vector<double>;

} // anonymous namespace

void bind_vector_arithmetic(py::class_<Vector>& cls)
{
    cls
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
        });
}

// =============================================================================
// Vector comparison
// =============================================================================


namespace
{

using Vector = stratax::container::Vector<double>;

} // anonymous namespace

void bind_vector_comparison(py::class_<Vector>& cls)
{
    cls
        .def("__eq__", [](const Vector& lhs, const Vector& rhs) {
            return lhs == rhs;
        })
        .def("__ne__", [](const Vector& lhs, const Vector& rhs) {
            return lhs != rhs;
        });
}

// =============================================================================
// Vector reshape
// =============================================================================


namespace
{

using Vector = stratax::container::Vector<double>;

}

void bind_vector_reshape(py::class_<Vector>& cls)
{
    cls
        .def(
            "reshape",
            [](const Vector& self,
               const stratax::core::Shape& shape)
            {
                return reshape(self, shape);
            },
            py::arg("shape"),
            "Return a reshaped tensor.")
        .def(
            "reshape",
            [](const Vector& self,
               const std::vector<std::size_t>& dims)
            {
                return reshape(self, stratax::core::Shape(dims));
            },
            py::arg("shape"),
            "Return a reshaped tensor.")
        .def(
            "flatten",
            [](const Vector& self)
            {
                return flatten(self);
            },
            "Return a flattened vector.");
}

// =============================================================================
// Vector registration
// =============================================================================

void bind_vector(py::module_& m)
{
    using Vector = stratax::container::Vector<double>;

    py::class_<Vector> cls(m, "Vector");

    bind_vector_constructors(cls);
    bind_vector_properties(cls);
    bind_vector_indexing(cls);
    bind_vector_arithmetic(cls);
    bind_vector_comparison(cls);
    bind_vector_reshape(cls);
}
