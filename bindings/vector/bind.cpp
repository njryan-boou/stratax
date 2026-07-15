#include <pybind11/pybind11.h>

#include <stratax/containers/Vector.hpp>

namespace py = pybind11;

void bind_vector_constructors(py::class_<stratax::container::Vector<double>>&);
void bind_vector_indexing(py::class_<stratax::container::Vector<double>>&);
void bind_vector_arithmetic(py::class_<stratax::container::Vector<double>>&);
void bind_vector_conversions(py::class_<stratax::container::Vector<double>>&);
void bind_vector_properties(py::class_<stratax::container::Vector<double>>&);

void bind_vector(py::module_& m)
{
    using Vector = stratax::container::Vector<double>;

    py::class_<Vector> cls(m, "Vector");

    bind_vector_constructors(cls);
    bind_vector_properties(cls);
    bind_vector_indexing(cls);
    bind_vector_arithmetic(cls);
    bind_vector_conversions(cls);
}
