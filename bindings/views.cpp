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
    py::class_<ArrayView>(m, "ArrayView")
        .def("__len__",
            [](const ArrayView& view)
            {
                return view.size();
            })

        .def_property_readonly(
            "size",
            &ArrayView::size)

        .def_property_readonly(
            "ndim",
            &ArrayView::ndim)

        .def("__getitem__",
    [](const ArrayView& view, py::object index) -> double
    {
        if (py::isinstance<py::tuple>(index))
        {
            const py::tuple indices =
                index.cast<py::tuple>();

            if (indices.size() != view.rank())
            {
                throw Exceptions::IndexError(
                    "ArrayView index rank must match view rank.");
            }

            std::vector<std::size_t> normalized;
            normalized.reserve(view.rank());

            for (std::size_t dim = 0; dim < view.rank(); ++dim)
            {
                const auto raw =
                    binding_utils::cast_index(
                        indices[dim],
                        "ArrayView indices must be integers.",
                        "ArrayView index is too large.");

                normalized.push_back(
                    stratax::indexing::normalize_index(
                        raw,
                        view.shape()[dim]));
            }

            const auto offset =
                stratax::indexing::offset(
                    view.strides(),
                    normalized);

            return view.data()[offset];
        }

        const auto raw =
            binding_utils::cast_index(
                index,
                "ArrayView index must be an integer.",
                "ArrayView index is too large.");

        return view.at(raw); // if you add at()
    })

        .def("__setitem__",
    [](ArrayView& view,
       py::object index,
       double value)
    {
        if (py::isinstance<py::tuple>(index))
        {
            const py::tuple indices =
                index.cast<py::tuple>();

            if (indices.size() != view.rank())
            {
                throw Exceptions::IndexError(
                    "ArrayView index rank must match view rank.");
            }

            std::vector<std::size_t> normalized;
            normalized.reserve(view.rank());

            for (std::size_t dim = 0; dim < view.rank(); ++dim)
            {
                const auto raw =
                    binding_utils::cast_index(
                        indices[dim],
                        "ArrayView indices must be integers.",
                        "ArrayView index is too large.");

                normalized.push_back(
                    stratax::indexing::normalize_index(
                        raw,
                        view.shape()[dim]));
            }

            const auto offset =
                stratax::indexing::offset(
                    view.strides(),
                    normalized);

            view.data()[offset] = value;
            return;
        }

        const auto raw =
            binding_utils::cast_index(
                index,
                "ArrayView index must be an integer.",
                "ArrayView index is too large.");

        // Flat indexing path.
        view[static_cast<std::size_t>(raw)] = value;
    });
}

} // namespace binding_utils
