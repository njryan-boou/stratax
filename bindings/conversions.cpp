#include <pybind11/pybind11.h>

#include <stratax/algorithms/Conversion.hpp>

namespace py = pybind11;

#define STRATAX_BIND_CONVERSION(NAME, DOC) \
    m.def(#NAME, [](const Vector& arr) { return stratax::conversion::NAME(arr); }, py::arg("arr"), DOC); \
    m.def(#NAME, [](const Matrix& arr) { return stratax::conversion::NAME(arr); }, py::arg("arr"), DOC); \
    m.def(#NAME, [](const Tensor& arr) { return stratax::conversion::NAME(arr); }, py::arg("arr"), DOC)

void bind_conversions(py::module_& m)
{
    using Vector = stratax::container::Vector<double>;
    using Matrix = stratax::container::Matrix<double>;
    using Tensor = stratax::container::Tensor<double>;

    STRATAX_BIND_CONVERSION(to_vector, "Convert an array-like object to a vector.");
    STRATAX_BIND_CONVERSION(to_matrix, "Convert an array-like object to a matrix.");
    STRATAX_BIND_CONVERSION(to_tensor, "Convert an array-like object to a tensor.");
}

#undef STRATAX_BIND_CONVERSION
