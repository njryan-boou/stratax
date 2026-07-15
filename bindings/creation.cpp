#include <pybind11/pybind11.h>

#include <stratax/containers/Creation.hpp>

namespace py = pybind11;

void bind_creation(py::module_& m)
{
    m.def(
        "zeros",
        [](const stratax::core::Shape& shape)
        {
            return creation::zeros<double>(shape);
        },
        py::arg("shape"),
        "Create a tensor filled with zeros.");

    m.def(
        "ones",
        [](const stratax::core::Shape& shape)
        {
            return creation::ones<double>(shape);
        },
        py::arg("shape"),
        "Create a tensor filled with ones.");

    m.def(
        "full",
        [](const stratax::core::Shape& shape, double value)
        {
            return creation::full<double>(shape, value);
        },
        py::arg("shape"),
        py::arg("value"),
        "Create a tensor filled with a constant value.");

    m.def(
        "identity",
        [](std::size_t size)
        {
            return creation::identity<double>(size);
        },
        py::arg("size"),
        "Create a square identity tensor.");
}