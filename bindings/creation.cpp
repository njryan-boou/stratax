#include <pybind11/pybind11.h>

#include <stratax/algorithms/Creation.hpp>
#include <stratax/core/Shape.hpp>

#include <cstddef>

namespace py = pybind11;

void bind_creation(py::module_& m)
{
    constexpr auto zeros_doc = "Create an array filled with zeros.";
    constexpr auto ones_doc = "Create an array filled with ones.";
    constexpr auto full_doc = "Create an array filled with a constant value.";

    m.def("zeros", [](const stratax::core::Shape& shape) { return stratax::creation::zeros<double>(shape); }, py::arg("shape"), zeros_doc);
    m.def("zeros", [](std::size_t size) { return stratax::creation::zeros<double>(size); }, py::arg("size"), zeros_doc);
    m.def("zeros", [](std::size_t rows, std::size_t cols) { return stratax::creation::zeros<double>(rows, cols); }, py::arg("rows"), py::arg("cols"), zeros_doc);
    m.def("ones", [](const stratax::core::Shape& shape) { return stratax::creation::ones<double>(shape); }, py::arg("shape"), ones_doc);
    m.def("ones", [](std::size_t size) { return stratax::creation::ones<double>(size); }, py::arg("size"), ones_doc);
    m.def("ones", [](std::size_t rows, std::size_t cols) { return stratax::creation::ones<double>(rows, cols); }, py::arg("rows"), py::arg("cols"), ones_doc);
    m.def("full", [](const stratax::core::Shape& shape, double value) { return stratax::creation::full<double>(shape, value); }, py::arg("shape"), py::arg("value"), full_doc);
    m.def("full", [](std::size_t size, double value) { return stratax::creation::full<double>(size, value); }, py::arg("size"), py::arg("value"), full_doc);
    m.def("full", [](std::size_t rows, std::size_t cols, double value) { return stratax::creation::full<double>(rows, cols, value); }, py::arg("rows"), py::arg("cols"), py::arg("value"), full_doc);

    m.def(
        "identity",
        [](std::size_t size) { return stratax::creation::identity<double>(size); },
        py::arg("size"),
        "Create a square identity matrix.");
}
