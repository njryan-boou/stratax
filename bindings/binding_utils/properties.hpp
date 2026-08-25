#pragma once

#include "utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/io/Print.hpp>

#include <algorithm>
#include <cstddef>
#include <sstream>

namespace binding_utils {

namespace py = pybind11;

template<typename Array>
void bind_properties(py::class_<Array>& cls)
{
    cls
        .def_property_readonly("size", [](const Array& arr) {
            return arr.size();
        })
        .def_property_readonly("rank", [](const Array& arr) {
            return arr.rank();
        })
        .def_property_readonly("empty", [](const Array& arr) {
            return arr.empty();
        })
        .def_property_readonly(
            "shape",
            [](const Array& arr) -> const auto& {
                return arr.shape();
            },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "strides",
            [](const Array& arr) -> const auto& {
                return arr.strides();
            },
            py::return_value_policy::reference_internal)
        .def("fill", [](Array& arr, const typename Array::value_type& value) {
            arr.fill(value);
        }, py::arg("value"))
        .def("tolist", [](const Array& arr) {
            return array_to_list(arr);
        })
        .def("__repr__", [](const Array& arr) {
            std::ostringstream os;
            os << arr;
            return os.str();
        });
}

template<typename Array>
void bind_length(py::class_<Array>& cls)
{
    cls.def("__len__", [](const Array& array) {
        return array.rank() == 0
            ? std::size_t{0}
            : array.shape()[0];
    });
}

template<typename Array>
void bind_iteration(py::class_<Array>& cls)
{
    cls.def(
        "__iter__",
        [](Array& array) {
            return py::make_iterator(
                array.begin(),
                array.end());
        },
        py::keep_alive<0, 1>());
}

template<typename Array>
void bind_contains(py::class_<Array>& cls)
{
    using value_type = typename Array::value_type;

    cls.def(
        "__contains__",
        [](const Array& array, const value_type& value) {
            return std::find(
                array.begin(),
                array.end(),
                value) != array.end();
        },
        py::arg("value"));
}

template<typename Array>
void bind_members(py::class_<Array>& cls)
{
    bind_properties(cls);
    bind_length(cls);
    bind_iteration(cls);
    bind_contains(cls);
}

} // namespace binding_utils
