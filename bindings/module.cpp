#include <pybind11/pybind11.h>

#include "utils.hpp"

#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Meta.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>

#include <cstddef>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace
{

using Shape = stratax::core::Shape;

void bind_exceptions(py::module_& m)
{
    py::object stratax_error = py::register_exception<Exceptions::StrataxError>(
        m,
        "StrataxError",
        PyExc_RuntimeError
    );
    py::register_exception<Exceptions::ShapeError>(m, "ShapeError", stratax_error.ptr());
    py::register_exception<Exceptions::DimensionError>(m, "DimensionError", stratax_error.ptr());
    py::register_exception<Exceptions::IndexError>(m, "IndexError", stratax_error.ptr());
    py::register_exception<Exceptions::TypeError>(m, "TypeError", stratax_error.ptr());
    py::register_exception<Exceptions::BroadcastError>(m, "BroadcastError", stratax_error.ptr());
    py::register_exception<Exceptions::ZeroDivisionError>(m, "ZeroDivisionError", stratax_error.ptr());
    py::register_exception<Exceptions::AxisError>(m, "AxisError", stratax_error.ptr());
}

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

} // anonymous namespace

void bind_vector(py::module_&);
void bind_matrix(py::module_&);
void bind_tensor(py::module_&);
void bind_conversions(py::module_&);
void bind_creation(py::module_&);
void bind_reductions(py::module_&);

PYBIND11_MODULE(_core, m)
{
    m.attr("__doc__") = STRATAX_PY_DOC;

    m.attr("__version__") = STRATAX_VERSION;

    m.attr("__author__") = STRATAX_AUTHOR;

    bind_exceptions(m);
    bind_shape(m);
    bind_vector(m);
    bind_matrix(m);
    bind_tensor(m);
    bind_conversions(m);
    bind_creation(m);
    bind_reductions(m);
}
