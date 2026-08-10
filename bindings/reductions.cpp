#include <pybind11/pybind11.h>

#include <stratax/algorithms/Reductions.hpp>

namespace py = pybind11;

void bind_reductions(py::module_& m)
{
    using Vector = stratax::container::Vector<double>;
    using Matrix = stratax::container::Matrix<double>;
    using Tensor = stratax::container::Tensor<double>;

#define STRATAX_BIND_REDUCTION(NAME) \
    m.def(#NAME, [](const Vector& arr) { return reduction::NAME(arr); }, py::arg("arr")); \
    m.def(#NAME, [](const Matrix& arr) { return reduction::NAME(arr); }, py::arg("arr")); \
    m.def(#NAME, [](const Tensor& arr) { return reduction::NAME(arr); }, py::arg("arr")); \
    m.def(#NAME, [](const Vector& arr, int axis, bool keepdims) { return reduction::NAME(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false); \
    m.def(#NAME, [](const Matrix& arr, int axis, bool keepdims) { return reduction::NAME(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false); \
    m.def(#NAME, [](const Tensor& arr, int axis, bool keepdims) { return reduction::NAME(arr, axis, keepdims); }, py::arg("arr"), py::arg("axis"), py::arg("keepdims") = false)

    STRATAX_BIND_REDUCTION(sum);
    STRATAX_BIND_REDUCTION(prod);
    STRATAX_BIND_REDUCTION(max);
    STRATAX_BIND_REDUCTION(min);
    STRATAX_BIND_REDUCTION(argmax);
    STRATAX_BIND_REDUCTION(argmin);
    STRATAX_BIND_REDUCTION(mean);
    STRATAX_BIND_REDUCTION(var);
    STRATAX_BIND_REDUCTION(std);

#undef STRATAX_BIND_REDUCTION
}