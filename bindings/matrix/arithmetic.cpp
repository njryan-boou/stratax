#include <pybind11/pybind11.h>

#include <stratax/containers/Matrix.hpp>
#include <stratax/ops/arithmetic.hpp>

namespace py = pybind11;

namespace
{

using Matrix = stratax::container::Matrix<double>;

} // anonymous namespace

void bind_matrix_arithmetic(py::class_<Matrix>& cls)
{
    cls
        .def("__add__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs + rhs;
        })
        .def("__add__", [](const Matrix& lhs, double rhs) {
            return lhs + rhs;
        })
        .def("__radd__", [](const Matrix& rhs, double lhs) {
            return lhs + rhs;
        })
        .def("__sub__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs - rhs;
        })
        .def("__sub__", [](const Matrix& lhs, double rhs) {
            return lhs - rhs;
        })
        .def("__rsub__", [](const Matrix& rhs, double lhs) {
            return lhs - rhs;
        })
        .def("__mul__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs * rhs;
        })
        .def("__mul__", [](const Matrix& lhs, double rhs) {
            return lhs * rhs;
        })
        .def("__rmul__", [](const Matrix& rhs, double lhs) {
            return lhs * rhs;
        })
        .def("__truediv__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs / rhs;
        })
        .def("__truediv__", [](const Matrix& lhs, double rhs) {
            return lhs / rhs;
        })
        .def("__rtruediv__", [](const Matrix& rhs, double lhs) {
            return lhs / rhs;
        })
        .def("__iadd__", [](Matrix& lhs, const Matrix& rhs) -> Matrix& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__iadd__", [](Matrix& lhs, double rhs) -> Matrix& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Matrix& lhs, const Matrix& rhs) -> Matrix& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Matrix& lhs, double rhs) -> Matrix& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Matrix& lhs, const Matrix& rhs) -> Matrix& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Matrix& lhs, double rhs) -> Matrix& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Matrix& lhs, const Matrix& rhs) -> Matrix& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Matrix& lhs, double rhs) -> Matrix& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__pos__", [](const Matrix& matrix) {
            return +matrix;
        })
        .def("__neg__", [](const Matrix& matrix) {
            return -matrix;
        });
}
