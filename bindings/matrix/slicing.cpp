#include <pybind11/pybind11.h>

#include "../common/utils.hpp"

#include <stratax/containers/Matrix.hpp>
#include <stratax/core/Exceptions.hpp>

#include <cstddef>

namespace py = pybind11;

namespace
{

using Matrix = stratax::container::Matrix<double>;

Matrix matrix_slice_runtime(
    const Matrix& matrix,
    const binding_utils::ResolvedSlice& rows,
    const binding_utils::ResolvedSlice& cols)
{
    Matrix result(
        static_cast<std::size_t>(rows.length),
        static_cast<std::size_t>(cols.length));

    py::ssize_t source_row = rows.start;
    for (std::size_t out_row = 0; out_row < result.rows(); ++out_row)
    {
        py::ssize_t source_col = cols.start;
        for (std::size_t out_col = 0; out_col < result.cols(); ++out_col)
        {
            result(out_row, out_col) = matrix(
                static_cast<std::size_t>(source_row),
                static_cast<std::size_t>(source_col));
            source_col += cols.step;
        }
        source_row += rows.step;
    }

    return result;
}

} // anonymous namespace

void bind_matrix_slicing(py::class_<Matrix>& cls)
{
    cls
        .def("__getitem__", [](const Matrix& matrix, py::object index) -> py::object {
            if (py::isinstance<py::slice>(index))
            {
                const auto row_slice = binding_utils::resolve_slice(
                    index.cast<py::slice>(),
                    matrix.rows(),
                    "Matrix row slice step cannot be zero.");

                const binding_utils::ResolvedSlice col_slice{
                    0,
                    1,
                    static_cast<py::ssize_t>(matrix.cols())
                };

                return py::cast(matrix_slice_runtime(matrix, row_slice, col_slice));
            }

            if (!py::isinstance<py::tuple>(index))
            {
                throw Exceptions::IndexError("Matrix index must be a (row, column) tuple.");
            }

            const py::tuple tuple_index = index.cast<py::tuple>();
            if (tuple_index.size() != 2) {
                throw Exceptions::IndexError("Matrix index must be a (row, column) tuple.");
            }

            const bool row_is_slice = py::isinstance<py::slice>(tuple_index[0]);
            const bool col_is_slice = py::isinstance<py::slice>(tuple_index[1]);

            if (!row_is_slice && !col_is_slice)
            {
                return py::cast(matrix.at(
                    binding_utils::resolve_index(
                        tuple_index[0],
                        matrix.rows(),
                        "Matrix row index must be an integer.",
                        "Matrix row index is too large to fit in a signed integer.",
                        "Matrix row index is out of bounds."),
                    binding_utils::resolve_index(
                        tuple_index[1],
                        matrix.cols(),
                        "Matrix column index must be an integer.",
                        "Matrix column index is too large to fit in a signed integer.",
                        "Matrix column index is out of bounds.")));
            }

            const auto row_slice = row_is_slice
                ? binding_utils::resolve_slice(tuple_index[0].cast<py::slice>(), matrix.rows(), "Matrix row slice step cannot be zero.")
                : binding_utils::single_index_slice(
                    tuple_index[0],
                    matrix.rows(),
                    "Matrix row index must be an integer.",
                    "Matrix row index is too large to fit in a signed integer.",
                    "Matrix row index is out of bounds.");

            const auto col_slice = col_is_slice
                ? binding_utils::resolve_slice(tuple_index[1].cast<py::slice>(), matrix.cols(), "Matrix column slice step cannot be zero.")
                : binding_utils::single_index_slice(
                    tuple_index[1],
                    matrix.cols(),
                    "Matrix column index must be an integer.",
                    "Matrix column index is too large to fit in a signed integer.",
                    "Matrix column index is out of bounds.");

            return py::cast(matrix_slice_runtime(matrix, row_slice, col_slice));
        });
}
