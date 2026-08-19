#pragma once

#include <pybind11/pybind11.h>

#include <stratax/ops/Comparison.hpp>

namespace binding_utils {

namespace py = pybind11;

#define STRATAX_BIND_COMPARISON(NAME, PYTHON_NAME, OPERATOR) \
    cls.def(#NAME, [](const Array& lhs, const Array& rhs) { return NAME(lhs, rhs); }, py::arg("other")); \
    cls.def(#NAME, [](const Array& lhs, double rhs) { return NAME(lhs, rhs); }, py::arg("other")); \
    cls.def(PYTHON_NAME, [](const Array& lhs, const Array& rhs) { return lhs OPERATOR rhs; }); \
    cls.def(PYTHON_NAME, [](const Array& lhs, double rhs) { return lhs OPERATOR rhs; })

template<typename Array>
void bind_comparison(py::class_<Array>& cls)
{
    STRATAX_BIND_COMPARISON(equal, "__eq__", ==);
    STRATAX_BIND_COMPARISON(not_equal, "__ne__", !=);
    STRATAX_BIND_COMPARISON(less, "__lt__", <);
    STRATAX_BIND_COMPARISON(less_equal, "__le__", <=);
    STRATAX_BIND_COMPARISON(greater, "__gt__", >);
    STRATAX_BIND_COMPARISON(greater_equal, "__ge__", >=);
}

#undef STRATAX_BIND_COMPARISON

#define STRATAX_BIND_ARRAY_COMPARISON(NAME, LEFT, RIGHT) \
    m.def(#NAME, [](const LEFT& lhs, const RIGHT& rhs) { return NAME(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"))

#define STRATAX_BIND_COMPARISON_FUNCTION(NAME) \
    STRATAX_BIND_ARRAY_COMPARISON(NAME, Vector, Vector); \
    STRATAX_BIND_ARRAY_COMPARISON(NAME, Vector, Matrix); \
    STRATAX_BIND_ARRAY_COMPARISON(NAME, Vector, Tensor); \
    STRATAX_BIND_ARRAY_COMPARISON(NAME, Matrix, Vector); \
    STRATAX_BIND_ARRAY_COMPARISON(NAME, Matrix, Matrix); \
    STRATAX_BIND_ARRAY_COMPARISON(NAME, Matrix, Tensor); \
    STRATAX_BIND_ARRAY_COMPARISON(NAME, Tensor, Vector); \
    STRATAX_BIND_ARRAY_COMPARISON(NAME, Tensor, Matrix); \
    STRATAX_BIND_ARRAY_COMPARISON(NAME, Tensor, Tensor); \
    m.def(#NAME, [](const Vector& lhs, double rhs) { return NAME(lhs, rhs); }, py::arg("lhs"), py::arg("rhs")); \
    m.def(#NAME, [](const Matrix& lhs, double rhs) { return NAME(lhs, rhs); }, py::arg("lhs"), py::arg("rhs")); \
    m.def(#NAME, [](const Tensor& lhs, double rhs) { return NAME(lhs, rhs); }, py::arg("lhs"), py::arg("rhs")); \
    m.def(#NAME, [](double lhs, const Vector& rhs) { return NAME(lhs, rhs); }, py::arg("lhs"), py::arg("rhs")); \
    m.def(#NAME, [](double lhs, const Matrix& rhs) { return NAME(lhs, rhs); }, py::arg("lhs"), py::arg("rhs")); \
    m.def(#NAME, [](double lhs, const Tensor& rhs) { return NAME(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"))

inline void bind_comparison_functions(py::module_& m)
{
    using Vector = stratax::container::Vector<double>;
    using Matrix = stratax::container::Matrix<double>;
    using Tensor = stratax::container::Tensor<double>;

    STRATAX_BIND_COMPARISON_FUNCTION(equal);
    STRATAX_BIND_COMPARISON_FUNCTION(not_equal);
    STRATAX_BIND_COMPARISON_FUNCTION(less);
    STRATAX_BIND_COMPARISON_FUNCTION(less_equal);
    STRATAX_BIND_COMPARISON_FUNCTION(greater);
    STRATAX_BIND_COMPARISON_FUNCTION(greater_equal);
}

#undef STRATAX_BIND_COMPARISON_FUNCTION
#undef STRATAX_BIND_ARRAY_COMPARISON

} // namespace binding_utils
