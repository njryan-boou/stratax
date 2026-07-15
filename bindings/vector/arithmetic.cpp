#include <pybind11/pybind11.h>

#include <stratax/containers/Vector.hpp>
#include <stratax/ops/arithmetic.hpp>

namespace py = pybind11;

namespace
{

using Vector = stratax::container::Vector<double>;

} // anonymous namespace

void bind_vector_arithmetic(py::class_<Vector>& cls)
{
    cls
        .def("__add__", [](const Vector& lhs, const Vector& rhs) {
            return lhs + rhs;
        })
        .def("__add__", [](const Vector& lhs, double rhs) {
            return lhs + rhs;
        })
        .def("__radd__", [](const Vector& rhs, double lhs) {
            return lhs + rhs;
        })
        .def("__sub__", [](const Vector& lhs, const Vector& rhs) {
            return lhs - rhs;
        })
        .def("__sub__", [](const Vector& lhs, double rhs) {
            return lhs - rhs;
        })
        .def("__rsub__", [](const Vector& rhs, double lhs) {
            return lhs - rhs;
        })
        .def("__mul__", [](const Vector& lhs, const Vector& rhs) {
            return lhs * rhs;
        })
        .def("__mul__", [](const Vector& lhs, double rhs) {
            return lhs * rhs;
        })
        .def("__rmul__", [](const Vector& rhs, double lhs) {
            return lhs * rhs;
        })
        .def("__truediv__", [](const Vector& lhs, const Vector& rhs) {
            return lhs / rhs;
        })
        .def("__truediv__", [](const Vector& lhs, double rhs) {
            return lhs / rhs;
        })
        .def("__rtruediv__", [](const Vector& rhs, double lhs) {
            return lhs / rhs;
        })
        .def("__iadd__", [](Vector& lhs, const Vector& rhs) -> Vector& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__iadd__", [](Vector& lhs, double rhs) -> Vector& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Vector& lhs, const Vector& rhs) -> Vector& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Vector& lhs, double rhs) -> Vector& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Vector& lhs, const Vector& rhs) -> Vector& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Vector& lhs, double rhs) -> Vector& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Vector& lhs, const Vector& rhs) -> Vector& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Vector& lhs, double rhs) -> Vector& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__pos__", [](const Vector& vector) {
            return +vector;
        })
        .def("__neg__", [](const Vector& vector) {
            return -vector;
        });
}
