#include <pybind11/pybind11.h>

#include <stratax/containers/Conversions.hpp>

namespace py = pybind11;

void bind_conversions(py::module_& m)
{
    using Vector = stratax::container::Vector<double>;
    using Matrix = stratax::container::Matrix<double>;
    using Tensor = stratax::container::Tensor<double>;

    m.def(
        "to_vector",
        [](const Vector& arr)
        {
            return to_vector(arr);
        },
        py::arg("arr"),
        "Convert an array-like object to a vector.");

    m.def(
        "to_vector",
        [](const Matrix& arr)
        {
            return to_vector(arr);
        },
        py::arg("arr"),
        "Convert an array-like object to a vector.");

    m.def(
        "to_vector",
        [](const Tensor& arr)
        {
            return to_vector(arr);
        },
        py::arg("arr"),
        "Convert an array-like object to a vector.");

    m.def(
        "to_matrix",
        [](const Vector& arr)
        {
            return to_matrix(arr);
        },
        py::arg("arr"),
        "Convert an array-like object to a matrix.");

    m.def(
        "to_matrix",
        [](const Matrix& arr)
        {
            return to_matrix(arr);
        },
        py::arg("arr"),
        "Convert an array-like object to a matrix.");

    m.def(
        "to_matrix",
        [](const Tensor& arr)
        {
            return to_matrix(arr);
        },
        py::arg("arr"),
        "Convert an array-like object to a matrix.");

    m.def(
        "to_tensor",
        [](const Vector& arr)
        {
            return to_tensor(arr);
        },
        py::arg("arr"),
        "Convert an array-like object to a tensor.");

    m.def(
        "to_tensor",
        [](const Matrix& arr)
        {
            return to_tensor(arr);
        },
        py::arg("arr"),
        "Convert an array-like object to a tensor.");

    m.def(
        "to_tensor",
        [](const Tensor& arr)
        {
            return to_tensor(arr);
        },
        py::arg("arr"),
        "Convert an array-like object to a tensor.");
}