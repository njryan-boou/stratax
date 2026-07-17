#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/core/containers/Tensor.hpp>
#include <stratax/core/algorithms/Reshape.hpp>
#include <stratax/core/containers/Shape.hpp>

#include <cstddef>
#include <vector>

namespace py = pybind11;

namespace
{

using Tensor = stratax::container::Tensor<double>;

}

void bind_tensor_reshape(py::class_<Tensor>& cls)
{
    cls
        .def(
            "reshape",
            [](const Tensor& self,
               const stratax::core::Shape& shape)
            {
                return reshape(self, shape);
            },
            py::arg("shape"),
            "Return a reshaped tensor.")
        .def(
            "reshape",
            [](const Tensor& self,
               const std::vector<std::size_t>& dims)
            {
                return reshape(self, stratax::core::Shape(dims));
            },
            py::arg("shape"),
            "Return a reshaped tensor.")
        .def(
            "flatten",
            [](const Tensor& self)
            {
                return flatten(self);
            },
            "Return a flattened vector.");
}
