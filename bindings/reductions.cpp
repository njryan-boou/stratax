#include <pybind11/pybind11.h>

#include <stratax/algorithms/Reductions.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

namespace py = pybind11;

void bind_reductions(py::module_& m)
{
    using Vector = stratax::container::Vector<double>;
    using Matrix = stratax::container::Matrix<double>;
    using Tensor = stratax::container::Tensor<double>;

    m.def("sum", [](const Vector& arr) { return reduction::sum(arr); }, py::arg("arr"));
    m.def("sum", [](const Matrix& arr) { return reduction::sum(arr); }, py::arg("arr"));
    m.def("sum", [](const Tensor& arr) { return reduction::sum(arr); }, py::arg("arr"));
    m.def("sum", [](const Vector& arr, int axis, bool keepdims) { return reduction::sum(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("sum", [](const Matrix& arr, int axis, bool keepdims) { return reduction::sum(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("sum", [](const Tensor& arr, int axis, bool keepdims) { return reduction::sum(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);

    m.def("prod", [](const Vector& arr) { return reduction::prod(arr); }, py::arg("arr"));
    m.def("prod", [](const Matrix& arr) { return reduction::prod(arr); }, py::arg("arr"));
    m.def("prod", [](const Tensor& arr) { return reduction::prod(arr); }, py::arg("arr"));
    m.def("prod", [](const Vector& arr, int axis, bool keepdims) { return reduction::prod(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("prod", [](const Matrix& arr, int axis, bool keepdims) { return reduction::prod(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("prod", [](const Tensor& arr, int axis, bool keepdims) { return reduction::prod(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);

    m.def("max", [](const Vector& arr) { return reduction::max(arr); }, py::arg("arr"));
    m.def("max", [](const Matrix& arr) { return reduction::max(arr); }, py::arg("arr"));
    m.def("max", [](const Tensor& arr) { return reduction::max(arr); }, py::arg("arr"));
    m.def("max", [](const Vector& arr, int axis, bool keepdims) { return reduction::max(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("max", [](const Matrix& arr, int axis, bool keepdims) { return reduction::max(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("max", [](const Tensor& arr, int axis, bool keepdims) { return reduction::max(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);

    m.def("min", [](const Vector& arr) { return reduction::min(arr); }, py::arg("arr"));
    m.def("min", [](const Matrix& arr) { return reduction::min(arr); }, py::arg("arr"));
    m.def("min", [](const Tensor& arr) { return reduction::min(arr); }, py::arg("arr"));
    m.def("min", [](const Vector& arr, int axis, bool keepdims) { return reduction::min(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("min", [](const Matrix& arr, int axis, bool keepdims) { return reduction::min(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("min", [](const Tensor& arr, int axis, bool keepdims) { return reduction::min(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);

    m.def("argmax", [](const Vector& arr) { return reduction::argmax(arr); }, py::arg("arr"));
    m.def("argmax", [](const Matrix& arr) { return reduction::argmax(arr); }, py::arg("arr"));
    m.def("argmax", [](const Tensor& arr) { return reduction::argmax(arr); }, py::arg("arr"));
    m.def("argmax", [](const Vector& arr, int axis, bool keepdims) { return reduction::argmax(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("argmax", [](const Matrix& arr, int axis, bool keepdims) { return reduction::argmax(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("argmax", [](const Tensor& arr, int axis, bool keepdims) { return reduction::argmax(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);

    m.def("argmin", [](const Vector& arr) { return reduction::argmin(arr); }, py::arg("arr"));
    m.def("argmin", [](const Matrix& arr) { return reduction::argmin(arr); }, py::arg("arr"));
    m.def("argmin", [](const Tensor& arr) { return reduction::argmin(arr); }, py::arg("arr"));
    m.def("argmin", [](const Vector& arr, int axis, bool keepdims) { return reduction::argmin(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("argmin", [](const Matrix& arr, int axis, bool keepdims) { return reduction::argmin(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("argmin", [](const Tensor& arr, int axis, bool keepdims) { return reduction::argmin(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);

    m.def("mean", [](const Vector& arr) { return reduction::mean(arr); }, py::arg("arr"));
    m.def("mean", [](const Matrix& arr) { return reduction::mean(arr); }, py::arg("arr"));
    m.def("mean", [](const Tensor& arr) { return reduction::mean(arr); }, py::arg("arr"));
    m.def("mean", [](const Vector& arr, int axis, bool keepdims) { return reduction::mean(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("mean", [](const Matrix& arr, int axis, bool keepdims) { return reduction::mean(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("mean", [](const Tensor& arr, int axis, bool keepdims) { return reduction::mean(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);

    m.def("var", [](const Vector& arr) { return reduction::var(arr); }, py::arg("arr"));
    m.def("var", [](const Matrix& arr) { return reduction::var(arr); }, py::arg("arr"));
    m.def("var", [](const Tensor& arr) { return reduction::var(arr); }, py::arg("arr"));
    m.def("var", [](const Vector& arr, int axis, bool keepdims) { return reduction::var(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("var", [](const Matrix& arr, int axis, bool keepdims) { return reduction::var(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("var", [](const Tensor& arr, int axis, bool keepdims) { return reduction::var(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);

    m.def("std", [](const Vector& arr) { return reduction::std(arr); }, py::arg("arr"));
    m.def("std", [](const Matrix& arr) { return reduction::std(arr); }, py::arg("arr"));
    m.def("std", [](const Tensor& arr) { return reduction::std(arr); }, py::arg("arr"));
    m.def("std", [](const Vector& arr, int axis, bool keepdims) { return reduction::std(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("std", [](const Matrix& arr, int axis, bool keepdims) { return reduction::std(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("std", [](const Tensor& arr, int axis, bool keepdims) { return reduction::std(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false);
}
