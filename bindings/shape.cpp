#include <pybind11/pybind11.h>

#include "utils.hpp"

#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/exceptions/Exceptions.hpp>

#include <cstddef>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace
{

using Shape = stratax::core::Shape;

long long cast_shape_dimension(py::handle dim)
{
    return binding_utils::cast_integer(
        dim,
        "Shape dimensions must be integers.",
        "Shape dimension is too large to fit in a signed integer.");
}

Shape make_shape(const std::vector<long long>& dims)
{
    std::vector<std::size_t> values;
    values.reserve(dims.size());
    std::size_t elements = dims.empty() ? 0 : 1;

    for (std::size_t i = 0; i < dims.size(); ++i)
    {
        const std::size_t value = stratax::core::validation::nonnegative_shape_dimension(
            dims[i],
            "Shape dimensions cannot be negative.");
        values.push_back(value);

        try
        {
            elements = stratax::core::validation::checked_multiply(
                elements,
                value,
                "Shape element count overflow.");
        }
        catch (const Exceptions::DimensionError& e)
        {
            binding_utils::raise_overflow(e.what());
        }
    }

    return Shape(values);
}

} // anonymous namespace

void bind_shape(py::module_& m)
{
    py::class_<Shape> cls(m, "Shape");

    cls
        .def(py::init<>())
        .def(py::init<const Shape&>(), py::arg("other"))
        .def(py::init([](py::int_ size)
        {
            return make_shape({cast_shape_dimension(size)});
        }), py::arg("size"))
        .def(py::init([](py::iterable dims)
        {
            std::vector<long long> values;

            for (py::handle dim : dims)
            {
                values.push_back(cast_shape_dimension(dim));
            }

            return make_shape(values);
        }), py::arg("dims"))
        .def_property_readonly("rank", &Shape::rank)
        .def_property_readonly("elements", [](const Shape& shape)
        {
            try
            {
                return shape.elements();
            }
            catch (const Exceptions::DimensionError& e)
            {
                binding_utils::raise_overflow(e.what());
            }
        })
        .def_property_readonly("empty", &Shape::empty)
        .def("__len__", &Shape::rank)
        .def("__getitem__", [](const Shape& shape, long long index)
        {
            return shape(binding_utils::normalize_index(
                index,
                shape.rank(),
                "Shape index is out of bounds."));
        })
        .def("__iter__", [](const Shape& shape)
        {
            return py::make_iterator(shape.begin(), shape.end());
        }, py::keep_alive<0, 1>())
        .def("__eq__", [](const Shape& lhs, const Shape& rhs)
        {
            return lhs == rhs;
        })
        .def("__ne__", [](const Shape& lhs, const Shape& rhs)
        {
            return lhs != rhs;
        })
        .def("__repr__", [](const Shape& shape)
        {
            std::ostringstream os;
            os << shape;
            return os.str();
        });
}