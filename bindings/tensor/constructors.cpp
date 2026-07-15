#include <pybind11/pybind11.h>

#include "../common/utils.hpp"

#include <stratax/containers/Tensor.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Validation.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace
{

using Tensor = stratax::container::Tensor<double>;
using Shape = stratax::core::Shape;

void require_allocatable_tensor_elements(std::size_t elements)
{
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        binding_utils::raise_overflow("Tensor storage size overflow.");
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
        binding_utils::raise_overflow(e.what());
    }
}

Shape make_shape(const std::vector<long long>& dims)
{
    std::vector<std::size_t> values;
    values.reserve(dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
        values.push_back(stratax::core::validation::nonnegative_shape_dimension(
            dims[i],
            "Tensor shape dimension cannot be negative."));
    }

    Shape shape(values);
    checked_tensor_elements(shape);
    return shape;
}

Shape make_shape(py::iterable dims)
{
    std::vector<long long> values;

    for (py::handle dim : dims) {
        values.push_back(binding_utils::cast_integer(
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

} // anonymous namespace

void bind_tensor_constructors(py::class_<Tensor>& cls)
{
    cls
        .def(py::init<>())
        .def(py::init([](const Shape& shape) {
            return make_tensor_from_shape(shape);
        }), py::arg("shape"))
        .def(py::init([](const Shape& shape, py::object value) {
            return make_tensor_from_shape(
                shape,
                binding_utils::cast_scalar(
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
                binding_utils::cast_scalar(
                    value,
                    "Tensor fill value must be a number.",
                    "Tensor fill value is too large to represent as a float."));
        }), py::arg("shape"), py::arg("value"));
}
