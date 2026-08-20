#pragma once

#include <pybind11/pybind11.h>

#include <stratax/core/ArrayView.hpp>

namespace py = pybind11;

namespace binding_utils {

struct PyArrayView
{
    stratax::core::ArrayView<double> view;
    py::object owner;

    PyArrayView(
        stratax::core::ArrayView<double> view,
        py::object owner)
        : view(std::move(view)),
          owner(std::move(owner))
    {}
};

void bind_array_view(pybind11::module_& m);

} // namespace binding_utils
