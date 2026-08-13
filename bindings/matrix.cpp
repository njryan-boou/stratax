#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.hpp"

#include <stratax/containers/Matrix.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/io/Print.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/indexing/Slicing.hpp>

#include "arithmetic.hpp"
#include "comparison.hpp"
#include "properties.hpp"
#include "reshape.hpp"

#include <cstddef>
#include <sstream>
#include <utility>
#include <vector>
#include <algorithm>

// Matrix constructors

namespace py = pybind11;

using Matrix = stratax::container::Matrix<double>;

namespace
{

Matrix make_matrix_from_iterable(py::iterable rows)
{
    std::vector<std::vector<double>> values;
    std::size_t cols = 0;
    bool saw_row = false;

    for (py::handle row_object : rows)
    {
        if (!py::isinstance<py::iterable>(row_object)
            || py::isinstance<py::str>(row_object))
        {
            throw Exceptions::TypeError("Matrix rows must be iterables of numbers.");
        }

        std::vector<double> row;
        for (py::handle value : row_object.cast<py::iterable>())
        {
            row.push_back(binding_utils::cast_scalar(
                value,
                "Matrix values must be numbers.",
                "Matrix value is too large to represent as a float."));
        }

        if (!saw_row)
        {
            cols = row.size();
            saw_row = true;
        }
        else if (row.size() != cols)
        {
            throw Exceptions::ShapeError("Matrix rows must all have the same length.");
        }

        values.push_back(std::move(row));
    }

    const std::size_t row_count = values.size();
    Matrix matrix(row_count, cols);

    auto dest = matrix.begin();

    for (const auto& row : values)
    {
        dest = std::copy(row.begin(), row.end(), dest);
    }

    return matrix;
}

} // anonymous namespace

void bind_matrix_constructors(py::class_<Matrix>& cls)
{
    cls
        .def(py::init<>())
        .def(py::init([](py::object rows, py::object cols) {
            const std::size_t row_count =
                stratax::core::validation::nonnegative_size(
                    binding_utils::cast_integer(
                        rows,
                        "Matrix row count must be an integer.",
                        "Matrix row count is too large to fit in a signed integer."),
                    "Matrix row count cannot be negative.");
            const std::size_t col_count =
                stratax::core::validation::nonnegative_size(
                    binding_utils::cast_integer(
                        cols,
                        "Matrix column count must be an integer.",
                        "Matrix column count is too large to fit in a signed integer."),
                    "Matrix column count cannot be negative.");
            return Matrix(row_count, col_count);
        }), py::arg("rows"), py::arg("cols"))
        .def(py::init<const Matrix&>(), py::arg("other"))
        .def(py::init([](py::object value) {
            if (!py::isinstance<py::iterable>(value) ||
                py::isinstance<py::str>(value))
            {
                throw Exceptions::TypeError(
                    "Matrix constructor expects an iterable of row values.");
            }

            return make_matrix_from_iterable(
                value.cast<py::iterable>());
        }), py::arg("value"))
        .def(py::init([](py::object rows, py::object cols, py::object value) {
                const std::size_t row_count =
                    stratax::core::validation::nonnegative_size(
                        binding_utils::cast_integer(
                            rows,
                            "Matrix row count must be an integer.",
                            "Matrix row count is too large to fit in a signed integer."),
                        "Matrix row count cannot be negative.");
                const std::size_t col_count =
                    stratax::core::validation::nonnegative_size(
                        binding_utils::cast_integer(
                            cols,
                            "Matrix column count must be an integer.",
                            "Matrix column count is too large to fit in a signed integer."),
                        "Matrix column count cannot be negative.");
            return Matrix(
                row_count,
                col_count,
                binding_utils::cast_scalar(
                    value,
                    "Matrix fill value must be a number.",
                    "Matrix fill value is too large to represent as a float."));
        }), py::arg("rows"), py::arg("cols"), py::arg("value"));
}

// Matrix properties

void bind_matrix_properties(py::class_<Matrix>& cls)
{
    bind_properties(cls);

    cls
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def("tolist", [](const Matrix& matrix) {
            std::vector<std::vector<double>> values(matrix.rows());
            for (std::size_t row = 0; row < matrix.rows(); ++row)
            {
                values[row].reserve(matrix.cols());
                for (std::size_t col = 0; col < matrix.cols(); ++col)
                {
                    values[row].push_back(matrix(row, col));
                }
            }
            return values;
        })
        .def("__repr__", [](const Matrix& matrix) {
            std::ostringstream os;
            os << matrix;
            return os.str();
        });
}

// Matrix indexing

void bind_matrix_indexing(py::class_<Matrix>& cls)
{
    cls
        .def("__len__", &Matrix::rows)
        .def("__getitem__", [](const Matrix& matrix, py::object index) -> py::object {
            if (py::isinstance<py::slice>(index))
            {
                const auto rows = binding_utils::cast_slice(
                    index.cast<py::slice>(),
                    matrix.rows());

                const stratax::core::Slice cols(
                    0,
                    static_cast<std::ptrdiff_t>(matrix.cols()));

                return py::cast(
                    slice(matrix, rows, cols));
            }

            if (!py::isinstance<py::tuple>(index))
            {
                throw Exceptions::IndexError("Matrix index must be a (row, column) tuple.");
            }

            const py::tuple tuple_index = index.cast<py::tuple>();
            if (tuple_index.size() != 2)
            {
                throw Exceptions::IndexError("Matrix index must be a (row, column) tuple.");
            }

            const bool row_is_slice = py::isinstance<py::slice>(tuple_index[0]);
            const bool col_is_slice = py::isinstance<py::slice>(tuple_index[1]);

            if (!row_is_slice && !col_is_slice)
            {
                return py::cast(matrix.at(
                    binding_utils::cast_index(
                        tuple_index[0],
                        "Matrix row index must be an integer.",
                        "Matrix row index is too large to fit in a signed integer."),
                    binding_utils::cast_index(
                        tuple_index[1],
                        "Matrix column index must be an integer.",
                        "Matrix column index is too large to fit in a signed integer.")));
            }

            const auto row_slice = row_is_slice
    ? binding_utils::cast_slice(
        tuple_index[0].cast<py::slice>(),
        matrix.rows())
    : binding_utils::single_index_slice(
        tuple_index[0],
        matrix.rows(),
        "Matrix row index must be an integer.",
        "Matrix row index is too large to fit in a signed integer.");

            const auto col_slice = col_is_slice
    ? binding_utils::cast_slice(
        tuple_index[1].cast<py::slice>(),
        matrix.cols())
    : binding_utils::single_index_slice(
        tuple_index[1],
        matrix.cols(),
        "Matrix column index must be an integer.",
        "Matrix column index is too large to fit in a signed integer.");

            return py::cast(
                slice(matrix, row_slice, col_slice));
        })
        .def("__setitem__", [](Matrix& matrix, py::tuple index, double value) {
            if (index.size() != 2)
            {
                throw Exceptions::IndexError("Matrix index must be a (row, column) tuple.");
            }

            matrix.at(
                binding_utils::cast_index(
                    index[0],
                    "Matrix row index must be an integer.",
                    "Matrix row index is too large to fit in a signed integer."),
                binding_utils::cast_index(
                    index[1],
                    "Matrix column index must be an integer.",
                    "Matrix column index is too large to fit in a signed integer.")
            ) = value;
        })
        .def("__iter__", [](const Matrix& matrix) {
            return py::make_iterator(matrix.begin(), matrix.end());
        }, py::keep_alive<0, 1>());
}

// Matrix registration

void bind_matrix(py::module_& m)
{
    py::class_<Matrix> cls(m, "Matrix");

    bind_matrix_constructors(cls);
    bind_matrix_properties(cls);
    bind_matrix_indexing(cls);
    bind_arithmetic(cls);
    bind_comparison(cls);
    binding_utils::bind_reshape(cls);
}