#include <pybind11/pybind11.h>

#include <stratax/containers/Matrix.hpp>

namespace py = pybind11;

void bind_matrix_constructors(py::class_<stratax::container::Matrix<double>>&);
void bind_matrix_indexing(py::class_<stratax::container::Matrix<double>>&);
void bind_matrix_arithmetic(py::class_<stratax::container::Matrix<double>>&);
void bind_matrix_slicing(py::class_<stratax::container::Matrix<double>>&);
void bind_matrix_conversions(py::class_<stratax::container::Matrix<double>>&);
void bind_matrix_properties(py::class_<stratax::container::Matrix<double>>&);

void bind_matrix(py::module_& m)
{
    using Matrix = stratax::container::Matrix<double>;

    py::class_<Matrix> cls(m, "Matrix");

    bind_matrix_constructors(cls);
    bind_matrix_properties(cls);
    bind_matrix_indexing(cls);
    bind_matrix_slicing(cls);
    bind_matrix_arithmetic(cls);
    bind_matrix_conversions(cls);
}
