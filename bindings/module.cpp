#include <pybind11/pybind11.h>

#include <stratax/core/Meta.hpp>

namespace py = pybind11;

void bind_common_exceptions(py::module_&);
void bind_common_helpers(py::module_&);
void bind_common_utilities(py::module_&);
void bind_shape(py::module_&);
void bind_vector(py::module_&);
void bind_matrix(py::module_&);
void bind_tensor(py::module_&);
void bind_conversions(py::module_&);
void bind_creation(py::module_&);

PYBIND11_MODULE(_core, m)
{
    m.attr("__doc__") = STRATAX_PY_DOC;

    m.attr("__version__") = STRATAX_VERSION;

    m.attr("__author__") = STRATAX_AUTHOR;

    bind_common_exceptions(m);
    bind_common_helpers(m);
    bind_common_utilities(m);

    bind_shape(m);
    bind_vector(m);
    bind_matrix(m);
    bind_tensor(m);
    bind_conversions(m);
    bind_creation(m);
}
