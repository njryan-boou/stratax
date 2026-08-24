#pragma once

#include <pybind11/pybind11.h>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/ops/Arithmetic.hpp>

namespace py = pybind11;

namespace binding_utils
{

using Vector = stratax::container::Vector<double>;
using Matrix = stratax::container::Matrix<double>;
using Tensor = stratax::container::Tensor<double>;

// -----------------------------------------------------------------------------
// Array-array arithmetic
// -----------------------------------------------------------------------------

template<typename L, typename R>
void bind_binary_arithmetic(py::class_<L>& cls)
{
    cls.def(
        "__add__",
        [](const L& lhs, const R& rhs)
        {
            return lhs + rhs;
        },
        py::is_operator());

    cls.def(
        "__sub__",
        [](const L& lhs, const R& rhs)
        {
            return lhs - rhs;
        },
        py::is_operator());

    cls.def(
        "__mul__",
        [](const L& lhs, const R& rhs)
        {
            return lhs * rhs;
        },
        py::is_operator());

    cls.def(
        "__truediv__",
        [](const L& lhs, const R& rhs)
        {
            return lhs / rhs;
        },
        py::is_operator());
}

// -----------------------------------------------------------------------------
// Array-array compound arithmetic
// -----------------------------------------------------------------------------

template<typename L, typename R>
void bind_compound_arithmetic(py::class_<L>& cls)
{
    cls.def(
        "__iadd__",
        [](L& lhs, const R& rhs) -> L&
        {
            return lhs += rhs;
        },
        py::return_value_policy::reference_internal,
        py::is_operator());

    cls.def(
        "__isub__",
        [](L& lhs, const R& rhs) -> L&
        {
            return lhs -= rhs;
        },
        py::return_value_policy::reference_internal,
        py::is_operator());

    cls.def(
        "__imul__",
        [](L& lhs, const R& rhs) -> L&
        {
            return lhs *= rhs;
        },
        py::return_value_policy::reference_internal,
        py::is_operator());

    cls.def(
        "__itruediv__",
        [](L& lhs, const R& rhs) -> L&
        {
            return lhs /= rhs;
        },
        py::return_value_policy::reference_internal,
        py::is_operator());
}

// -----------------------------------------------------------------------------
// Array-scalar arithmetic
// -----------------------------------------------------------------------------

template<typename A>
void bind_scalar_arithmetic(py::class_<A>& cls)
{
    using Scalar = typename A::value_type;

    cls.def(
        "__add__",
        [](const A& lhs, const Scalar& rhs)
        {
            return lhs + rhs;
        },
        py::is_operator());

    cls.def(
        "__sub__",
        [](const A& lhs, const Scalar& rhs)
        {
            return lhs - rhs;
        },
        py::is_operator());

    cls.def(
        "__mul__",
        [](const A& lhs, const Scalar& rhs)
        {
            return lhs * rhs;
        },
        py::is_operator());

    cls.def(
        "__truediv__",
        [](const A& lhs, const Scalar& rhs)
        {
            return lhs / rhs;
        },
        py::is_operator());
}

// -----------------------------------------------------------------------------
// Scalar-array arithmetic
// -----------------------------------------------------------------------------

template<typename A>
void bind_reverse_scalar_arithmetic(py::class_<A>& cls)
{
    using Scalar = typename A::value_type;

    cls.def(
        "__radd__",
        [](const A& rhs, const Scalar& lhs)
        {
            return lhs + rhs;
        },
        py::is_operator());

    cls.def(
        "__rsub__",
        [](const A& rhs, const Scalar& lhs)
        {
            return lhs - rhs;
        },
        py::is_operator());

    cls.def(
        "__rmul__",
        [](const A& rhs, const Scalar& lhs)
        {
            return lhs * rhs;
        },
        py::is_operator());

    cls.def(
        "__rtruediv__",
        [](const A& rhs, const Scalar& lhs)
        {
            return lhs / rhs;
        },
        py::is_operator());
}

// -----------------------------------------------------------------------------
// Array-scalar compound arithmetic
// -----------------------------------------------------------------------------

template<typename A>
void bind_compound_scalar_arithmetic(py::class_<A>& cls)
{
    using Scalar = typename A::value_type;

    cls.def(
        "__iadd__",
        [](A& lhs, const Scalar& rhs) -> A&
        {
            return lhs += rhs;
        },
        py::return_value_policy::reference_internal,
        py::is_operator());

    cls.def(
        "__isub__",
        [](A& lhs, const Scalar& rhs) -> A&
        {
            return lhs -= rhs;
        },
        py::return_value_policy::reference_internal,
        py::is_operator());

    cls.def(
        "__imul__",
        [](A& lhs, const Scalar& rhs) -> A&
        {
            return lhs *= rhs;
        },
        py::return_value_policy::reference_internal,
        py::is_operator());

    cls.def(
        "__itruediv__",
        [](A& lhs, const Scalar& rhs) -> A&
        {
            return lhs /= rhs;
        },
        py::return_value_policy::reference_internal,
        py::is_operator());
}

// -----------------------------------------------------------------------------
// Unary arithmetic
// -----------------------------------------------------------------------------

template<typename A>
void bind_unary_arithmetic(py::class_<A>& cls)
{
    cls.def(
        "__neg__",
        [](const A& arr)
        {
            return -arr;
        },
        py::is_operator());

    cls.def(
        "__pos__",
        [](const A& arr)
        {
            return +arr;
        },
        py::is_operator());
}

// -----------------------------------------------------------------------------
// One RHS container
// -----------------------------------------------------------------------------

template<typename L, typename R>
void bind_array_arithmetic(py::class_<L>& cls)
{
    bind_binary_arithmetic<L, R>(cls);
    bind_compound_arithmetic<L, R>(cls);
}

// -----------------------------------------------------------------------------
// Complete arithmetic API
// -----------------------------------------------------------------------------

template<typename A>
void bind_arithmetic(py::class_<A>& cls)
{
    // Every Python container may operate with every other Python container.
    bind_array_arithmetic<A, Vector>(cls);
    bind_array_arithmetic<A, Matrix>(cls);
    bind_array_arithmetic<A, Tensor>(cls);

    // Scalar arithmetic.
    bind_scalar_arithmetic(cls);
    bind_reverse_scalar_arithmetic(cls);
    bind_compound_scalar_arithmetic(cls);

    // Unary arithmetic.
    bind_unary_arithmetic(cls);
}

} // namespace binding_utils