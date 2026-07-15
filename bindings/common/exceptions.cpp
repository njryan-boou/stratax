#include <pybind11/pybind11.h>

#include <stratax/core/Exceptions.hpp>

namespace py = pybind11;

void bind_common_exceptions(py::module_& m)
{
    py::object stratax_error = py::register_exception<Exceptions::StrataxError>(
        m,
        "StrataxError",
        PyExc_RuntimeError
    );
    py::register_exception<Exceptions::ShapeError>(m, "ShapeError", stratax_error.ptr());
    py::register_exception<Exceptions::DimensionError>(m, "DimensionError", stratax_error.ptr());
    py::register_exception<Exceptions::IndexError>(m, "IndexError", stratax_error.ptr());
    py::register_exception<Exceptions::TypeError>(m, "TypeError", stratax_error.ptr());
    py::register_exception<Exceptions::BroadcastError>(m, "BroadcastError", stratax_error.ptr());
    py::register_exception<Exceptions::ZeroDivisionError>(m, "ZeroDivisionError", stratax_error.ptr());
}
