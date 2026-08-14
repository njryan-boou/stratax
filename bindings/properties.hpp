#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

template<typename Array>
void bind_properties(py::class_<Array>& cls)
{
    cls
        .def_property_readonly("size", &Array::size)
        .def_property_readonly("rank", &Array::rank)
        .def_property_readonly("empty", &Array::empty)
        .def_property_readonly(
            "shape",
            &Array::shape,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "strides",
            [](const Array& arr) {
                return std::vector<std::size_t>(
                    arr.strides().begin(),
                    arr.strides().end());
            })
        .def("fill", &Array::fill, py::arg("value"));
}