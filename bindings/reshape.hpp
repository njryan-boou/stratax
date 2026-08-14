#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/algorithms/Reshape.hpp>
#include <stratax/core/Shape.hpp>

#include <cstddef>
#include <vector>

namespace py = pybind11;

template<typename Array>
void bind_reshape(py::class_<Array>& cls)
{
    cls
        .def(
            "reshape",
            [](const Array& self, const stratax::core::Shape& shape) {
                return reshape(self, shape);
            },
            py::arg("shape")
        )
        .def(
            "reshape",
            [](const Array& self, const std::vector<std::size_t>& dims) {
                return reshape(
                    self,
                    stratax::core::Shape(dims)
                );
            },
            py::arg("shape")
        )
        .def(
            "flatten",
            [](const Array& self) {
                return flatten(self);
            }
        );
}