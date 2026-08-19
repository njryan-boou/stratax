#include <pybind11/pybind11.h>

#include <stratax/exceptions/Exceptions.hpp>

#include "comparison.hpp"

namespace py = pybind11;

namespace
{

constexpr const char* PyDoc = "Python bindings for Stratax";
constexpr const char* PyVersion = "0.3.1";
constexpr const char* PyAuthor = "Noah Ryan";
constexpr const char* PyLicense = "MIT";

void bind_exceptions(py::module_& m)
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
    py::register_exception<Exceptions::AxisError>(m, "AxisError", stratax_error.ptr());
}

} // anonymous namespace

void bind_shape(py::module_&);
void bind_boolean_arrays(py::module_&);
void bind_vector(py::module_&);
void bind_matrix(py::module_&);
void bind_tensor(py::module_&);
void bind_conversions(py::module_&);
void bind_creation(py::module_&);
void bind_reductions(py::module_&);

PYBIND11_MODULE(_core, m)
{
    m.attr("__doc__") = PyDoc;
    m.attr("__version__") = PyVersion;
    m.attr("__author__") = PyAuthor;
    m.attr("__license__") = PyLicense;

    bind_exceptions(m);
    bind_shape(m);
    bind_boolean_arrays(m);
    bind_vector(m);
    bind_matrix(m);
    bind_tensor(m);
    bind_comparison_functions(m);
    bind_conversions(m);
    bind_creation(m);
    bind_reductions(m);
}
