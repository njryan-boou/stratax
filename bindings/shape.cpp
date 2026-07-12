#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Validation.hpp>

#include <cstddef>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

[[noreturn]] void raise_overflow(const char* message)
{
    PyErr_SetString(PyExc_OverflowError, message);
    throw py::error_already_set();
}

long long cast_shape_dimension(py::handle dim)
{
    if (py::isinstance<py::bool_>(dim) || !py::isinstance<py::int_>(dim))
    {
        throw Exceptions::TypeError(
            "Shape dimensions must be integers."
        );
    }

    const long long value = PyLong_AsLongLong(dim.ptr());
    if (PyErr_Occurred())
    {
        PyErr_Clear();
        raise_overflow("Shape dimension is too large to fit in a signed integer.");
    }

    return value;
}

stratax::core::Shape make_shape(const std::vector<long long>& dims)
{
    stratax::core::Buffer<std::size_t> buffer(dims.size());
    std::size_t elements = dims.empty() ? 0 : 1;

    for (std::size_t i = 0; i < dims.size(); ++i)
    {
        buffer[i] = stratax::core::validation::nonnegative_shape_dimension(
            dims[i],
            "Shape dimensions cannot be negative.");

        try
        {
            elements = stratax::core::validation::checked_multiply(
                elements,
                buffer[i],
                "Shape element count overflow.");
        }
        catch (const Exceptions::DimensionError& e)
        {
            raise_overflow(e.what());
        }
    }

    return stratax::core::Shape(std::move(buffer));
}

}

void bind_shape(py::module_& m)
{
    using Shape = stratax::core::Shape;

    py::class_<Shape>(m, "_Shape")

        // Constructors
        .def(py::init<>())

        .def(py::init<const Shape&>(),
            py::arg("other"))

        .def(py::init([](py::iterable dims)
        {
            std::vector<long long> values;

            for (py::handle dim : dims)
            {
                values.push_back(cast_shape_dimension(dim));
            }

            return make_shape(values);
        }),
        py::arg("dims"))

        // Methods
        .def("rank", &Shape::rank)

        .def("elements", [](const Shape& shape)
        {
            try
            {
                return shape.elements();
            }
            catch (const Exceptions::DimensionError& e)
            {
                raise_overflow(e.what());
            }
        })

        .def("empty", &Shape::empty)

        // Python special methods
        .def("__len__", &Shape::rank)

        .def("__getitem__",
            [](const Shape& shape, long long index)
            {
                return shape(stratax::core::validation::nonnegative_index(
                    index,
                    "Shape index cannot be negative."));
            })

        .def("__iter__",
            [](const Shape& shape)
            {
                return py::make_iterator(
                    shape.begin(),
                    shape.end()
                );
            },
            py::keep_alive<0, 1>())

        .def("__eq__",
            [](const Shape& lhs, const Shape& rhs)
            {
                return lhs == rhs;
            })

        .def("__ne__",
            [](const Shape& lhs, const Shape& rhs)
            {
                return lhs != rhs;
            })

        .def("__repr__",
            [](const Shape& shape)
            {
                std::ostringstream os;
                os << shape;
                return os.str();
            });
}
