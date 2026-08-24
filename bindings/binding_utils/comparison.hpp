#pragma once

#include <pybind11/pybind11.h>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/ops/Comparison.hpp>

#include <type_traits>

namespace binding_utils {

namespace py = pybind11;

template<typename Array, typename Other>
void bind_array_comparison_methods(py::class_<Array>& cls)
{
    cls
        .def(
            "equal",
            [](const Array& lhs, const Other& rhs) {
                return equal(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "not_equal",
            [](const Array& lhs, const Other& rhs) {
                return not_equal(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "less",
            [](const Array& lhs, const Other& rhs) {
                return less(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "less_equal",
            [](const Array& lhs, const Other& rhs) {
                return less_equal(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "greater",
            [](const Array& lhs, const Other& rhs) {
                return greater(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "greater_equal",
            [](const Array& lhs, const Other& rhs) {
                return greater_equal(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "__eq__",
            [](const Array& lhs, const Other& rhs) {
                return lhs == rhs;
            },
            py::is_operator())
        .def(
            "__ne__",
            [](const Array& lhs, const Other& rhs) {
                return lhs != rhs;
            },
            py::is_operator())
        .def(
            "__lt__",
            [](const Array& lhs, const Other& rhs) {
                return lhs < rhs;
            },
            py::is_operator())
        .def(
            "__le__",
            [](const Array& lhs, const Other& rhs) {
                return lhs <= rhs;
            },
            py::is_operator())
        .def(
            "__gt__",
            [](const Array& lhs, const Other& rhs) {
                return lhs > rhs;
            },
            py::is_operator())
        .def(
            "__ge__",
            [](const Array& lhs, const Other& rhs) {
                return lhs >= rhs;
            },
            py::is_operator());
}

template<typename Array>
void bind_comparison(py::class_<Array>& cls)
{
    using Vector = stratax::container::Vector<double>;
    using Matrix = stratax::container::Matrix<double>;
    using Tensor = stratax::container::Tensor<double>;

    bind_array_comparison_methods<Array, Array>(cls);

    if constexpr (!std::is_same_v<Array, Vector>)
    {
        bind_array_comparison_methods<Array, Vector>(cls);
    }
    if constexpr (!std::is_same_v<Array, Matrix>)
    {
        bind_array_comparison_methods<Array, Matrix>(cls);
    }
    if constexpr (!std::is_same_v<Array, Tensor>)
    {
        bind_array_comparison_methods<Array, Tensor>(cls);
    }

    cls
        .def(
            "equal",
            [](const Array& lhs, double rhs) {
                return equal(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "not_equal",
            [](const Array& lhs, double rhs) {
                return not_equal(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "less",
            [](const Array& lhs, double rhs) {
                return less(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "less_equal",
            [](const Array& lhs, double rhs) {
                return less_equal(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "greater",
            [](const Array& lhs, double rhs) {
                return greater(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "greater_equal",
            [](const Array& lhs, double rhs) {
                return greater_equal(lhs, rhs);
            },
            py::arg("other"))
        .def(
            "__eq__",
            [](const Array& lhs, double rhs) {
                return lhs == rhs;
            },
            py::is_operator())
        .def(
            "__ne__",
            [](const Array& lhs, double rhs) {
                return lhs != rhs;
            },
            py::is_operator())
        .def(
            "__lt__",
            [](const Array& lhs, double rhs) {
                return lhs < rhs;
            },
            py::is_operator())
        .def(
            "__le__",
            [](const Array& lhs, double rhs) {
                return lhs <= rhs;
            },
            py::is_operator())
        .def(
            "__gt__",
            [](const Array& lhs, double rhs) {
                return lhs > rhs;
            },
            py::is_operator())
        .def(
            "__ge__",
            [](const Array& lhs, double rhs) {
                return lhs >= rhs;
            },
            py::is_operator());
}

template<typename Left, typename Right, typename Comparison>
void bind_array_comparison(
    py::module_& m,
    const char* name,
    Comparison comparison)
{
    m.def(name, [comparison](const Left& lhs, const Right& rhs) { return comparison(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));
}

template<typename Vector, typename Matrix, typename Tensor, typename Comparison>
void bind_comparison_function(
    py::module_& m,
    const char* name,
    Comparison comparison)
{
    bind_array_comparison<Vector, Vector>(m, name, comparison);
    bind_array_comparison<Vector, Matrix>(m, name, comparison);
    bind_array_comparison<Vector, Tensor>(m, name, comparison);
    bind_array_comparison<Matrix, Vector>(m, name, comparison);
    bind_array_comparison<Matrix, Matrix>(m, name, comparison);
    bind_array_comparison<Matrix, Tensor>(m, name, comparison);
    bind_array_comparison<Tensor, Vector>(m, name, comparison);
    bind_array_comparison<Tensor, Matrix>(m, name, comparison);
    bind_array_comparison<Tensor, Tensor>(m, name, comparison);
    m.def(name, [comparison](const Vector& lhs, double rhs) { return comparison(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));
    m.def(name, [comparison](const Matrix& lhs, double rhs) { return comparison(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));
    m.def(name, [comparison](const Tensor& lhs, double rhs) { return comparison(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));
    m.def(name, [comparison](double lhs, const Vector& rhs) { return comparison(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));
    m.def(name, [comparison](double lhs, const Matrix& rhs) { return comparison(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));
    m.def(name, [comparison](double lhs, const Tensor& rhs) { return comparison(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));
}

inline void bind_comparison_functions(py::module_& m)
{
    using Vector = stratax::container::Vector<double>;
    using Matrix = stratax::container::Matrix<double>;
    using Tensor = stratax::container::Tensor<double>;

    bind_comparison_function<Vector, Matrix, Tensor>(m, "equal", [](const auto& lhs, const auto& rhs) { return equal(lhs, rhs); });
    bind_comparison_function<Vector, Matrix, Tensor>(m, "not_equal", [](const auto& lhs, const auto& rhs) { return not_equal(lhs, rhs); });
    bind_comparison_function<Vector, Matrix, Tensor>(m, "less", [](const auto& lhs, const auto& rhs) { return less(lhs, rhs); });
    bind_comparison_function<Vector, Matrix, Tensor>(m, "less_equal", [](const auto& lhs, const auto& rhs) { return less_equal(lhs, rhs); });
    bind_comparison_function<Vector, Matrix, Tensor>(m, "greater", [](const auto& lhs, const auto& rhs) { return greater(lhs, rhs); });
    bind_comparison_function<Vector, Matrix, Tensor>(m, "greater_equal", [](const auto& lhs, const auto& rhs) { return greater_equal(lhs, rhs); });
}

} // namespace binding_utils
