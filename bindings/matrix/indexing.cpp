#include <pybind11/pybind11.h>

#include "../common/utils.hpp"

#include <stratax/containers/Matrix.hpp>
#include <stratax/core/Exceptions.hpp>

namespace py = pybind11;

namespace
{

using Matrix = stratax::container::Matrix<double>;

} // anonymous namespace

void bind_matrix_indexing(py::class_<Matrix>& cls)
{
    cls
        .def("__len__", &Matrix::rows)
        .def("__setitem__", [](Matrix& matrix, py::tuple index, double value) {
            if (index.size() != 2) {
                throw Exceptions::IndexError("Matrix index must be a (row, column) tuple.");
            }

            matrix.at(
                binding_utils::resolve_index(
                    index[0],
                    matrix.rows(),
                    "Matrix row index must be an integer.",
                    "Matrix row index is too large to fit in a signed integer.",
                    "Matrix row index is out of bounds."),
                binding_utils::resolve_index(
                    index[1],
                    matrix.cols(),
                    "Matrix column index must be an integer.",
                    "Matrix column index is too large to fit in a signed integer.",
                    "Matrix column index is out of bounds.")
            ) = value;
        })
        .def("__iter__", [](const Matrix& matrix) {
            return py::make_iterator(matrix.begin(), matrix.end());
        }, py::keep_alive<0, 1>());
}
