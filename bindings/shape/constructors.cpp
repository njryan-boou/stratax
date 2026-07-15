#include <pybind11/pybind11.h>

#include "../common/utils.hpp"

#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Validation.hpp>

#include <cstddef>
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

void bind_shape_constructors(py::class_<Shape>& cls)
{
    cls
        .def(py::init<>())
        .def(py::init<const Shape&>(), py::arg("other"))
        .def(py::init([](py::iterable dims)
        {
            std::vector<long long> values;

            for (py::handle dim : dims)
            {
                values.push_back(cast_shape_dimension(dim));
            }

            return make_shape(values);
        }), py::arg("dims"));
}
