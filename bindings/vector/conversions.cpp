#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/containers/Vector.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/ops/reshape.hpp>

#include <cstddef>
#include <vector>

namespace py = pybind11;

namespace
{

using Vector = stratax::container::Vector<double>;

} // anonymous namespace

void bind_vector_conversions(py::class_<Vector>& cls)
{
    cls
        .def(
            "reshape",
            [](const Vector& self,
               const stratax::core::Shape& shape)
            {
                return reshape(self, shape);
            },
            py::arg("shape"),
            "Return a reshaped tensor.")
        .def(
            "reshape",
            [](const Vector& self,
               const std::vector<std::size_t>& dims)
            {
                return reshape(self, stratax::core::Shape(dims));
            },
            py::arg("shape"),
            "Return a reshaped tensor.")
        .def(
            "flatten",
            [](const Vector& self)
            {
                return flatten(self);
            },
            "Return a flattened vector.");
}
