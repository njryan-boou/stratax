#include <pybind11/pybind11.h>

#include <stratax/algorithms/Conversion.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

namespace py = pybind11;

void bind_conversions(py::module_& m)
{
    using Vector = stratax::container::Vector<double>;
    using Matrix = stratax::container::Matrix<double>;
    using Tensor = stratax::container::Tensor<double>;

    constexpr auto vector_doc = "Convert an array-like object to a vector.";
    constexpr auto matrix_doc = "Convert an array-like object to a matrix.";
    constexpr auto tensor_doc = "Convert an array-like object to a tensor.";

    m.def("to_vector", [](const Vector& arr) { return stratax::conversion::to_vector(arr); }, py::arg("arr"), vector_doc);
    m.def("to_vector", [](const Matrix& arr) { return stratax::conversion::to_vector(arr); }, py::arg("arr"), vector_doc);
    m.def("to_vector", [](const Tensor& arr) { return stratax::conversion::to_vector(arr); }, py::arg("arr"), vector_doc);
    m.def("to_matrix", [](const Vector& arr) { return stratax::conversion::to_matrix(arr); }, py::arg("arr"), matrix_doc);
    m.def("to_matrix", [](const Matrix& arr) { return stratax::conversion::to_matrix(arr); }, py::arg("arr"), matrix_doc);
    m.def("to_matrix", [](const Tensor& arr) { return stratax::conversion::to_matrix(arr); }, py::arg("arr"), matrix_doc);
    m.def("to_tensor", [](const Vector& arr) { return stratax::conversion::to_tensor(arr); }, py::arg("arr"), tensor_doc);
    m.def("to_tensor", [](const Matrix& arr) { return stratax::conversion::to_tensor(arr); }, py::arg("arr"), tensor_doc);
    m.def("to_tensor", [](const Tensor& arr) { return stratax::conversion::to_tensor(arr); }, py::arg("arr"), tensor_doc);
}
