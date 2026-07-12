#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/containers/Tensor.hpp>
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
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Tensor = stratax::container::Tensor<double>;
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

void require_allocatable_tensor_elements(std::size_t elements)
{
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        raise_overflow("Tensor storage size overflow.");
    }
}

std::size_t checked_tensor_elements(const Shape& shape)
{
    try
    {
        const std::size_t elements = shape.elements();
        require_allocatable_tensor_elements(elements);
        return elements;
    }
    catch (const Exceptions::DimensionError& e)
    {
        raise_overflow(e.what());
    }
}

std::size_t tensor_offset(const stratax::container::Tensor<double>& tensor, py::tuple index)
{
    const auto& shape = tensor.shape();
    if (index.size() != shape.rank()) {
        throw Exceptions::IndexError("Tensor index rank must match tensor rank.");
    }

    std::size_t offset = 0;
    std::size_t stride = 1;

    for (std::size_t reverse_dim = shape.rank(); reverse_dim > 0; --reverse_dim) {
        const std::size_t dim = reverse_dim - 1;
        const std::size_t component = stratax::core::validation::nonnegative_index(
            cast_integer(index[dim], "Tensor index components must be integers.", "Tensor index component is too large to fit in a signed integer."),
            "Tensor index component cannot be negative."
        );

        if (component >= shape(dim)) {
            throw Exceptions::IndexError("Tensor index component is out of bounds.");
        }

        try
        {
            const std::size_t component_offset =
                stratax::core::validation::checked_multiply(
                    component,
                    stride,
                    "Tensor index offset multiplication overflow.");

            offset = stratax::core::validation::checked_add(
                offset,
                component_offset,
                "Tensor index offset addition overflow.");

            stride = stratax::core::validation::checked_multiply(
                stride,
                shape(dim),
                "Tensor index stride overflow.");
        }
        catch (const Exceptions::DimensionError& e)
        {
            raise_overflow(e.what());
        }
    }

    return offset;
}

Shape make_shape(const std::vector<std::size_t>& dims)
{
    stratax::core::Buffer<std::size_t> buffer(dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
        buffer[i] = dims[i];
    }

    Shape shape(std::move(buffer));
    checked_tensor_elements(shape);
    return shape;
}

Shape make_shape(const std::vector<long long>& dims)
{
    stratax::core::Buffer<std::size_t> buffer(dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
        buffer[i] = stratax::core::validation::nonnegative_shape_dimension(
            dims[i],
            "Tensor shape dimension cannot be negative.");
    }

    Shape shape(std::move(buffer));
    checked_tensor_elements(shape);
    return shape;
}

Shape make_shape(py::iterable dims)
{
    std::vector<long long> values;

    for (py::handle dim : dims) {
        values.push_back(cast_integer(
            dim,
            "Tensor shape dimensions must be integers.",
            "Tensor shape dimension is too large to fit in a signed integer."));
    }

    return make_shape(values);
}

Shape make_shape_from_object(py::object dims)
{
    if (py::isinstance<py::iterable>(dims) && !py::isinstance<py::str>(dims))
    {
        return make_shape(dims.cast<py::iterable>());
    }

    throw Exceptions::TypeError("Tensor shape must be a Shape or iterable of dimensions.");
}

Tensor make_tensor_from_shape(const Shape& shape)
{
    checked_tensor_elements(shape);
    return Tensor(shape);
}

Tensor make_tensor_from_shape(const Shape& shape, double value)
{
    checked_tensor_elements(shape);
    return Tensor(shape, value);
}

}

void bind_tensor(py::module_& m)
{
    py::class_<Tensor>(m, "Tensor")
        // Constructors
        .def(py::init<>())
        .def(py::init([](const Shape& shape) {
            return make_tensor_from_shape(shape);
        }), py::arg("shape"))
        .def(py::init([](const Shape& shape, py::object value) {
            return make_tensor_from_shape(
                shape,
                cast_scalar(
                    value,
                    "Tensor fill value must be a number.",
                    "Tensor fill value is too large to represent as a float."));
        }), py::arg("shape"), py::arg("value"))
        .def(py::init<const Tensor&>(), py::arg("other"))
        .def(py::init([](py::object dims) {
            return Tensor(make_shape_from_object(dims));
        }), py::arg("shape"))
        .def(py::init([](py::object dims, py::object value) {
            return Tensor(
                make_shape_from_object(dims),
                cast_scalar(
                    value,
                    "Tensor fill value must be a number.",
                    "Tensor fill value is too large to represent as a float."));
        }), py::arg("shape"), py::arg("value"))

        // Methods
        .def("size", &Tensor::size)
        .def("rank", &Tensor::rank)
        .def("empty", &Tensor::empty)
        .def("shape", &Tensor::shape, py::return_value_policy::reference_internal)
        .def("strides", [](const Tensor& tensor) {
            std::vector<std::size_t> values;
            const auto& strides = tensor.strides();
            values.reserve(strides.rank());
            for (std::size_t stride : strides) {
                values.push_back(stride);
            }
            return values;
        })
        .def("fill", &Tensor::fill, py::arg("value"))
        .def("tolist", [](const Tensor& tensor) {
            std::vector<double> values;
            values.reserve(tensor.size());
            for (std::size_t i = 0; i < tensor.size(); ++i) {
                values.push_back(tensor[i]);
            }
            return values;
        })

        // Python special methods
        .def("__len__", &Tensor::size)
        .def("__getitem__", [](const Tensor& tensor, py::object index) {
            if (py::isinstance<py::tuple>(index)) {
                return tensor.at(tensor_offset(tensor, index.cast<py::tuple>()));
            }

            return tensor.at(
                stratax::core::validation::nonnegative_index(
                    cast_integer(index, "Tensor index must be an integer.", "Tensor index is too large to fit in a signed integer."),
                    "Tensor index cannot be negative.")
            );
        })
        .def("__setitem__", [](Tensor& tensor, py::object index, double value) {
            if (py::isinstance<py::tuple>(index)) {
                tensor.at(tensor_offset(tensor, index.cast<py::tuple>())) = value;
                return;
            }

            tensor.at(
                stratax::core::validation::nonnegative_index(
                    cast_integer(index, "Tensor index must be an integer.", "Tensor index is too large to fit in a signed integer."),
                    "Tensor index cannot be negative.")
            ) = value;
        })
        .def("__iter__", [](const Tensor& tensor) {
            return py::make_iterator(tensor.begin(), tensor.end());
        }, py::keep_alive<0, 1>())
        .def("__eq__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs == rhs;
        })
        .def("__ne__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs != rhs;
        })
        .def("__add__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs + rhs;
        })
        .def("__add__", [](const Tensor& lhs, double rhs) {
            return lhs + rhs;
        })
        .def("__radd__", [](const Tensor& rhs, double lhs) {
            return lhs + rhs;
        })
        .def("__sub__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs - rhs;
        })
        .def("__sub__", [](const Tensor& lhs, double rhs) {
            return lhs - rhs;
        })
        .def("__rsub__", [](const Tensor& rhs, double lhs) {
            return lhs - rhs;
        })
        .def("__mul__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs * rhs;
        })
        .def("__mul__", [](const Tensor& lhs, double rhs) {
            return lhs * rhs;
        })
        .def("__rmul__", [](const Tensor& rhs, double lhs) {
            return lhs * rhs;
        })
        .def("__truediv__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs / rhs;
        })
        .def("__truediv__", [](const Tensor& lhs, double rhs) {
            return lhs / rhs;
        })
        .def("__rtruediv__", [](const Tensor& rhs, double lhs) {
            return lhs / rhs;
        })
        .def("__iadd__", [](Tensor& lhs, const Tensor& rhs) -> Tensor& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__iadd__", [](Tensor& lhs, double rhs) -> Tensor& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Tensor& lhs, const Tensor& rhs) -> Tensor& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Tensor& lhs, double rhs) -> Tensor& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Tensor& lhs, const Tensor& rhs) -> Tensor& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Tensor& lhs, double rhs) -> Tensor& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Tensor& lhs, const Tensor& rhs) -> Tensor& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Tensor& lhs, double rhs) -> Tensor& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__pos__", [](const Tensor& tensor) {
            return +tensor;
        })
        .def("__neg__", [](const Tensor& tensor) {
            return -tensor;
        })
        .def("__repr__", [](const Tensor& tensor) {
            std::ostringstream os;
            os << tensor;
            return os.str();
        });
}
