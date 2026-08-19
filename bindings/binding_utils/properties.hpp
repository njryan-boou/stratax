#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

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
        }, py::arg("value"));
}

} // namespace binding_utils
