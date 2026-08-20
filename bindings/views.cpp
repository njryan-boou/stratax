#include "binding_utils/views.hpp"
#include "binding_utils/utils.hpp"

#include <cstddef>

#include <pybind11/pybind11.h>

#include <stratax/core/ArrayView.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace py = pybind11;

namespace binding_utils {

namespace {

using ArrayView = stratax::core::ArrayView<double>;

} // namespace

void bind_array_view(py::module_& m)
{
    py::class_<PyArrayView>(m, "ArrayView")
        .def("__len__",
            [](const PyArrayView& self)
            {
                return self.view.size();
            })

        .def_property_readonly(
            "size",
            [](const PyArrayView& self)
            {
                return self.view.size();
            })

        .def_property_readonly(
            "ndim",
            [](const PyArrayView& self)
            {
                return self.view.ndim();
            })

        .def("__getitem__",
            [](const PyArrayView& self, py::object index)
            {
                if (py::isinstance<py::tuple>(index))
                {
                    const py::tuple tuple =
                        index.cast<py::tuple>();

                    std::vector<std::ptrdiff_t> indices;
                    indices.reserve(tuple.size());

                    for (const auto& item : tuple)
                    {
                        indices.push_back(
                            py::cast<std::ptrdiff_t>(item));
                    }

                    return self.view.at(indices);
                }

                return self.view.at(
                    py::cast<std::ptrdiff_t>(index));
            })

        .def("__setitem__",
            [](PyArrayView& self,
               py::object index,
               double value)
            {
                if (py::isinstance<py::tuple>(index))
                {
                    const py::tuple tuple =
                        index.cast<py::tuple>();

                    std::vector<std::ptrdiff_t> indices;
                    indices.reserve(tuple.size());

                    for (const auto& item : tuple)
                    {
                        indices.push_back(
                            py::cast<std::ptrdiff_t>(item));
                    }

                    self.view.at(indices) = value;
                    return;
                }

                self.view.at(
                    py::cast<std::ptrdiff_t>(index)) = value;
            });
}

} // namespace binding_utils
