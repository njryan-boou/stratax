#include "binding_utils/views.hpp"
#include "binding_utils/utils.hpp"

#include <stratax/io/Print.hpp>

#include <cstddef>
#include <sstream>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace binding_utils {

void bind_array_view(py::module_& m)
{
    py::class_<PyArrayView>(m, "ArrayView")
        .def(
            "__len__",
            [](const PyArrayView& self) {
                return self.view.size();
            })
        .def_property_readonly(
            "size",
            [](const PyArrayView& self) {
                return self.view.size();
            })
        .def_property_readonly(
            "ndim",
            [](const PyArrayView& self) {
                return self.view.ndim();
            })
        .def_property_readonly("rank", [](const PyArrayView& self) { return self.view.rank(); })
        .def_property_readonly("empty", [](const PyArrayView& self) { return self.view.empty(); })
        .def_property_readonly("shape", [](const PyArrayView& self) { return self.view.shape(); })
        .def_property_readonly("strides", [](const PyArrayView& self) { return self.view.strides(); })
        .def(
            "tolist",
            [](const PyArrayView& self) {
                return array_to_list(self.view);
            })
        .def(
            "__repr__",
            [](const PyArrayView& self) {
                std::ostringstream os;
                os << self.view;
                return os.str();
            })
        .def(
            "__getitem__",
            [](const PyArrayView& self, py::object index) {
                if (py::isinstance<py::tuple>(index))
                {
                    const py::tuple tuple =
                        index.cast<py::tuple>();

                    std::vector<std::ptrdiff_t> indices;
                    indices.reserve(tuple.size());

                    for (const auto& item : tuple)
                    {
                        indices.push_back(
                            cast_index(item));
                    }

                    return self.view.at(indices);
                }

                return self.view.at(
                    cast_index(index));
            })
        .def(
            "__setitem__",
            [](PyArrayView& self, py::object index, py::object value) {
                if (py::isinstance<py::tuple>(index))
                {
                    const py::tuple tuple =
                        index.cast<py::tuple>();

                    std::vector<std::ptrdiff_t> indices;
                    indices.reserve(tuple.size());

                    for (const auto& item : tuple)
                    {
                        indices.push_back(
                            cast_index(item));
                    }

                    self.view.at(indices) = cast_scalar(value);
                    return;
                }

                self.view.at(
                    cast_index(index)) = cast_scalar(value);
            });
}

} // namespace binding_utils
