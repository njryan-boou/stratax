#include <pybind11/pybind11.h>

#include "../common/utils.hpp"

#include <stratax/containers/Vector.hpp>

#include <cstddef>

namespace py = pybind11;

namespace
{

using Vector = stratax::container::Vector<double>;

Vector slice_vector_runtime(const Vector& vector, const binding_utils::ResolvedSlice& slice)
{
    Vector result(static_cast<std::size_t>(slice.length));

    py::ssize_t source = slice.start;
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result[i] = vector[static_cast<std::size_t>(source)];
        source += slice.step;
    }

    return result;
}

} // anonymous namespace

void bind_vector_indexing(py::class_<Vector>& cls)
{
    cls
        .def("__len__", &Vector::size)
        .def("__getitem__", [](const Vector& vector, py::object index) -> py::object {
            if (py::isinstance<py::slice>(index))
            {
                const auto range = binding_utils::resolve_slice(
                    index.cast<py::slice>(),
                    vector.size(),
                    "Vector slice step cannot be zero.");
                return py::cast(slice_vector_runtime(vector, range));
            }

            return py::cast(vector.at(binding_utils::resolve_index(
                index,
                vector.size(),
                "Vector index must be an integer.",
                "Vector index is too large to fit in a signed integer.",
                "Vector index is out of bounds.")));
        })
        .def("__setitem__", [](Vector& vector, py::object index, double value) {
            vector.at(binding_utils::resolve_index(
                index,
                vector.size(),
                "Vector index must be an integer.",
                "Vector index is too large to fit in a signed integer.",
                "Vector index is out of bounds.")) = value;
        })
        .def("__iter__", [](const Vector& vector) {
            return py::make_iterator(vector.begin(), vector.end());
        }, py::keep_alive<0, 1>());
}
