#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/containers/Matrix.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Slicing.hpp>

#include "binding_utils/arithmetic.hpp"
#include "binding_utils/comparison.hpp"
#include "binding_utils/properties.hpp"
#include "binding_utils/reshape.hpp"
#include "binding_utils/views.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

// Matrix constructors

namespace py = pybind11;
using namespace binding_utils;

using Matrix = stratax::container::Matrix<double>;

namespace
{

std::size_t checked_matrix_dimension(long long value, bool rows)
{
    if (value < 0)
    {
        throw Exceptions::DimensionError(
            rows ? "Matrix row count cannot be negative."
                 : "Matrix column count cannot be negative.");
    }

    return checked_size(value);
}

void ensure_matrix_storage_fits(std::size_t rows, std::size_t cols)
{
    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols)
    {
        raise_overflow(
            Exceptions::OverflowError("Matrix element count overflow."));
    }

    const std::size_t elements = rows * cols;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        raise_overflow(
            Exceptions::OverflowError("Matrix storage size overflow."));
    }
}

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
            throw Exceptions::TypeError(
                "Matrix rows must be iterables of numbers.");
        }

        std::vector<double> row;
        for (py::handle value : row_object.cast<py::iterable>())
        {
            row.push_back(cast_scalar(
                value));
        }

        if (!saw_row)
        {
            cols = row.size();
            saw_row = true;
        }
        else if (row.size() != cols)
        {
            throw Exceptions::ShapeError(
                "Matrix rows must have equal lengths.");
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
            const std::size_t row_count = checked_matrix_dimension(
                cast_integer(rows),
                true);
            const std::size_t col_count = checked_matrix_dimension(
                cast_integer(cols),
                false);
            ensure_matrix_storage_fits(row_count, col_count);
            return Matrix(row_count, col_count);
        }), py::arg("rows"), py::arg("cols"))
        .def(py::init<const stratax::core::Shape&>(), py::arg("shape"))
        .def(py::init<const Matrix&>(), py::arg("other"))
        .def(py::init([](py::object value) {
            if (!py::isinstance<py::iterable>(value) ||
                py::isinstance<py::str>(value))
            {
                throw Exceptions::TypeError(
                    "Matrix constructor expects an iterable of rows.");
            }

            return make_matrix_from_iterable(
                value.cast<py::iterable>());
        }), py::arg("value"))
        .def(py::init([](py::object rows, py::object cols, py::object value) {
            const std::size_t row_count = checked_matrix_dimension(
                cast_integer(rows),
                true);
            const std::size_t col_count = checked_matrix_dimension(
                cast_integer(cols),
                false);
            ensure_matrix_storage_fits(row_count, col_count);
            return Matrix(
                row_count,
                col_count,
                cast_scalar(
                    value));
        }), py::arg("rows"), py::arg("cols"), py::arg("value"));
}

// Matrix properties

void bind_matrix_properties(py::class_<Matrix>& cls)
{
    bind_properties(cls);

    cls
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols);
}

// Matrix indexing

void bind_matrix_indexing(py::class_<Matrix>& cls)
{
    cls
        .def(
            "__getitem__",
            [](py::object self, py::object index) -> py::object {
                Matrix& matrix = self.cast<Matrix&>();

                if (py::isinstance<py::slice>(index))
                {
                    index = py::make_tuple(index, py::slice(py::none(), py::none(), py::none()));
                }

                if (!py::isinstance<py::tuple>(index))
                {
                    return py::cast(
                        matrix.at(
                            cast_index(index)));
                }

                const py::tuple tuple_index =
                    index.cast<py::tuple>();

                if (tuple_index.size() != 2)
                {
                    throw Exceptions::IndexError(
                        "Matrix indexing requires exactly two indices.");
                }

                const bool row_slice =
                    py::isinstance<py::slice>(tuple_index[0]);
                const bool col_slice =
                    py::isinstance<py::slice>(tuple_index[1]);

                if (!row_slice && !col_slice)
                {
                    const auto row =
                        cast_index(tuple_index[0]);
                    const auto col =
                        cast_index(tuple_index[1]);

                    return py::cast(matrix.at(row, col));
                }

                const stratax::core::Slice rows = row_slice
                    ? cast_slice(
                        tuple_index[0].cast<py::slice>(),
                        matrix.rows())
                    : single_index_slice(
                        tuple_index[0],
                        matrix.rows());

                const stratax::core::Slice cols = col_slice
                    ? cast_slice(
                        tuple_index[1].cast<py::slice>(),
                        matrix.cols())
                    : single_index_slice(
                        tuple_index[1],
                        matrix.cols());

                auto view = stratax::indexing::slice(
                    matrix,
                    rows,
                    cols);

                return py::cast(
                    PyArrayView(
                        std::move(view),
                        self));
            })
        .def(
            "__setitem__",
            [](Matrix& matrix, py::tuple index, double value) {
                if (index.size() != 2)
                {
                    throw Exceptions::IndexError(
                        "Matrix assignment requires exactly two indices.");
                }

                matrix.at(
                    cast_index(index[0]),
                    cast_index(index[1])) = value;
            });
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
    bind_reshape(cls);
    bind_members(cls);
}
