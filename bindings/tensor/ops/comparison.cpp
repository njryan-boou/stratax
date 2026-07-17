#include <pybind11/pybind11.h>

#include <stratax/core/containers/Tensor.hpp>
#include <stratax/core/ops/Comparison.hpp>

namespace py = pybind11;

namespace
{

using Tensor = stratax::container::Tensor<double>;

} // anonymous namespace

void bind_tensor_comparison(py::class_<Tensor>& cls)
{
    cls
        .def("__eq__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs == rhs;
        })
        .def("__ne__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs != rhs;
        });
}
