#pragma once

#include <pybind11/pybind11.h>

#include <stratax/ops/Comparison.hpp>

namespace py = pybind11;

template<typename Array>
void bind_comparison(py::class_<Array>& cls)
{
    cls
        .def("__eq__", [](const Array& lhs, const Array& rhs) {
            return lhs == rhs;
        })
        .def("__ne__", [](const Array& lhs, const Array& rhs) {
            return lhs != rhs;
        });
};