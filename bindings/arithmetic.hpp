#pragma once

#include <pybind11/pybind11.h>

#include <stratax/ops/Arithmetic.hpp>

namespace py = pybind11;

#define STRATAX_BIND_BINARY_ARITHMETIC(NAME, REVERSE_NAME, OPERATOR) \
    cls.def(NAME, [](const Array& lhs, const Array& rhs) { return lhs OPERATOR rhs; }, py::is_operator()); \
    cls.def(NAME, [](const Array& lhs, double rhs) { return lhs OPERATOR rhs; }, py::is_operator()); \
    cls.def(REVERSE_NAME, [](const Array& rhs, double lhs) { return lhs OPERATOR rhs; }, py::is_operator())

#define STRATAX_BIND_INPLACE_ARITHMETIC(NAME, OPERATOR) \
    cls.def(NAME, [](Array& lhs, const Array& rhs) -> Array& { lhs OPERATOR rhs; return lhs; }, py::return_value_policy::reference_internal, py::is_operator()); \
    cls.def(NAME, [](Array& lhs, double rhs) -> Array& { lhs OPERATOR rhs; return lhs; }, py::return_value_policy::reference_internal, py::is_operator())

template<typename Array>
void bind_arithmetic(py::class_<Array>& cls)
{
    STRATAX_BIND_BINARY_ARITHMETIC("__add__", "__radd__", +);
    STRATAX_BIND_BINARY_ARITHMETIC("__sub__", "__rsub__", -);
    STRATAX_BIND_BINARY_ARITHMETIC("__mul__", "__rmul__", *);
    STRATAX_BIND_BINARY_ARITHMETIC("__truediv__", "__rtruediv__", /);

    STRATAX_BIND_INPLACE_ARITHMETIC("__iadd__", +=);
    STRATAX_BIND_INPLACE_ARITHMETIC("__isub__", -=);
    STRATAX_BIND_INPLACE_ARITHMETIC("__imul__", *=);
    STRATAX_BIND_INPLACE_ARITHMETIC("__itruediv__", /=);

    cls
        .def("__pos__", [](const Array& arr) { return +arr; }, py::is_operator())
        .def("__neg__", [](const Array& arr) { return -arr; }, py::is_operator());
}

#undef STRATAX_BIND_INPLACE_ARITHMETIC
#undef STRATAX_BIND_BINARY_ARITHMETIC
