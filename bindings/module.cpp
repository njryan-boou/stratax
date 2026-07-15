#include <pybind11/pybind11.h>

#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Meta.hpp>

namespace py = pybind11;

void bind_shape(py::module_&);
void bind_vector(py::module_&);
void bind_matrix(py::module_&);
void bind_tensor(py::module_&);

PYBIND11_MODULE(_core, m)
{
    m.attr("__doc__") = STRATAX_PY_DOC;

    m.attr("__version__") = STRATAX_VERSION;

    m.attr("__author__") = STRATAX_AUTHOR;

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

    bind_shape(m);
    bind_vector(m);
    bind_matrix(m);
    bind_tensor(m);
}
