#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

template<typename Array>
void bind_arithmetic(py::class_<Array>& cls)
{
    cls
        .def("__add__", [](const Array& lhs, const Array& rhs) {
            return lhs + rhs;
        })
        .def("__add__", [](const Array& lhs, double rhs) {
            return lhs + rhs;
        })
        .def("__radd__", [](const Array& rhs, double lhs) {
            return lhs + rhs;
        })
        .def("__sub__", [](const Array& lhs, const Array& rhs) {
            return lhs - rhs;
        })
        .def("__sub__", [](const Array& lhs, double rhs) {
            return lhs - rhs;
        })
        .def("__rsub__", [](const Array& rhs, double lhs) {
            return lhs - rhs;
        })
        .def("__mul__", [](const Array& lhs, const Array& rhs) {
            return lhs * rhs;
        })
        .def("__mul__", [](const Array& lhs, double rhs) {
            return lhs * rhs;
        })
        .def("__rmul__", [](const Array& rhs, double lhs) {
            return lhs * rhs;
        })
        .def("__truediv__", [](const Array& lhs, const Array& rhs) {
            return lhs / rhs;
        })
        .def("__truediv__", [](const Array& lhs, double rhs) {
            return lhs / rhs;
        })
        .def("__rtruediv__", [](const Array& rhs, double lhs) {
            return lhs / rhs;
        })
        .def("__iadd__", [](Array& lhs, const Array& rhs) -> Array& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__iadd__", [](Array& lhs, double rhs) -> Array& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Array& lhs, const Array& rhs) -> Array& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Array& lhs, double rhs) -> Array& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Array& lhs, const Array& rhs) -> Array& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Array& lhs, double rhs) -> Array& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Array& lhs, const Array& rhs) -> Array& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Array& lhs, double rhs) -> Array& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__pos__", [](const Array& arr) {
            return +arr;
        })
        .def("__neg__", [](const Array& arr) {
            return -arr;
        });
};