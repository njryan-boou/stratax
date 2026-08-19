#include <pybind11/pybind11.h>

#include <stratax/algorithms/Creation.hpp>

namespace py = pybind11;

#define STRATAX_BIND_CONSTANT_CREATION(NAME, DOC) \
    m.def(#NAME, [](const stratax::core::Shape& shape) { return stratax::creation::NAME<double>(shape); }, py::arg("shape"), DOC); \
    m.def(#NAME, [](std::size_t size) { return stratax::creation::NAME<double>(size); }, py::arg("size"), DOC); \
    m.def(#NAME, [](std::size_t rows, std::size_t cols) { return stratax::creation::NAME<double>(rows, cols); }, py::arg("rows"), py::arg("cols"), DOC)

#define STRATAX_BIND_FULL(DOC) \
    m.def("full", [](const stratax::core::Shape& shape, double value) { return stratax::creation::full<double>(shape, value); }, py::arg("shape"), py::arg("value"), DOC); \
    m.def("full", [](std::size_t size, double value) { return stratax::creation::full<double>(size, value); }, py::arg("size"), py::arg("value"), DOC); \
    m.def("full", [](std::size_t rows, std::size_t cols, double value) { return stratax::creation::full<double>(rows, cols, value); }, py::arg("rows"), py::arg("cols"), py::arg("value"), DOC)

void bind_creation(py::module_& m)
{
    STRATAX_BIND_CONSTANT_CREATION(zeros, "Create an array filled with zeros.");
    STRATAX_BIND_CONSTANT_CREATION(ones, "Create an array filled with ones.");
    STRATAX_BIND_FULL("Create an array filled with a constant value.");

    m.def(
        "identity",
        [](std::size_t size) { return stratax::creation::identity<double>(size); },
        py::arg("size"),
        "Create a square identity matrix.");
}

#undef STRATAX_BIND_FULL
#undef STRATAX_BIND_CONSTANT_CREATION
