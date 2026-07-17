#include <pybind11/pybind11.h>

#include <stratax/core/containers/Tensor.hpp>
#include <stratax/core/ops/Arithmetic.hpp>

namespace py = pybind11;

namespace
{

using Tensor = stratax::container::Tensor<double>;

} // anonymous namespace

void bind_tensor_arithmetic(py::class_<Tensor>& cls)
{
    cls
        .def("__add__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs + rhs;
        })
        .def("__add__", [](const Tensor& lhs, double rhs) {
            return lhs + rhs;
        })
        .def("__radd__", [](const Tensor& rhs, double lhs) {
            return lhs + rhs;
        })
        .def("__sub__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs - rhs;
        })
        .def("__sub__", [](const Tensor& lhs, double rhs) {
            return lhs - rhs;
        })
        .def("__rsub__", [](const Tensor& rhs, double lhs) {
            return lhs - rhs;
        })
        .def("__mul__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs * rhs;
        })
        .def("__mul__", [](const Tensor& lhs, double rhs) {
            return lhs * rhs;
        })
        .def("__rmul__", [](const Tensor& rhs, double lhs) {
            return lhs * rhs;
        })
        .def("__truediv__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs / rhs;
        })
        .def("__truediv__", [](const Tensor& lhs, double rhs) {
            return lhs / rhs;
        })
        .def("__rtruediv__", [](const Tensor& rhs, double lhs) {
            return lhs / rhs;
        })
        .def("__iadd__", [](Tensor& lhs, const Tensor& rhs) -> Tensor& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__iadd__", [](Tensor& lhs, double rhs) -> Tensor& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Tensor& lhs, const Tensor& rhs) -> Tensor& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Tensor& lhs, double rhs) -> Tensor& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Tensor& lhs, const Tensor& rhs) -> Tensor& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Tensor& lhs, double rhs) -> Tensor& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Tensor& lhs, const Tensor& rhs) -> Tensor& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Tensor& lhs, double rhs) -> Tensor& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__pos__", [](const Tensor& tensor) {
            return +tensor;
        })
        .def("__neg__", [](const Tensor& tensor) {
            return -tensor;
        });
}
