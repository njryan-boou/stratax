#include <pybind11/pybind11.h>

#include "../common/utils.hpp"

#include <stratax/containers/Matrix.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Validation.hpp>

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

using Matrix = stratax::container::Matrix<double>;
using Shape = stratax::core::Shape;

void require_allocatable_matrix_elements(std::size_t elements)
{
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        binding_utils::raise_overflow("Matrix storage size overflow.");
    }
}

std::size_t checked_matrix_elements(std::size_t rows, std::size_t cols)
{
    try
    {
        const std::size_t elements = stratax::core::validation::checked_multiply(
            rows,
            cols,
            "Matrix element count overflow.");
        require_allocatable_matrix_elements(elements);
        return elements;
    }
    catch (const Exceptions::DimensionError& e)
    {
        binding_utils::raise_overflow(e.what());
    }
}

std::size_t checked_matrix_dimension(long long value, const char* message)
{
    return stratax::core::validation::nonnegative_size(value, message);
}

Matrix make_matrix_from_shape(const Shape& shape)
{
    stratax::core::validation::require_rank(shape, 2, "Matrix shape must have rank 2.");
    checked_matrix_elements(shape(0), shape(1));
    return Matrix(shape);
}

Matrix make_matrix_from_iterable(py::iterable rows)
{
    std::vector<std::vector<double>> values;
    std::size_t cols = 0;
    bool saw_row = false;

    for (py::handle row_object : rows)
    {
        if (!py::isinstance<py::iterable>(row_object) || py::isinstance<py::str>(row_object))
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
    checked_matrix_elements(row_count, cols);
    Matrix matrix(row_count, cols);

    for (std::size_t row = 0; row < row_count; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            matrix(row, col) = values[row][col];
        }
    }

    return matrix;
}

Matrix make_matrix_from_object(py::object value)
{
    if (py::isinstance<py::iterable>(value) && !py::isinstance<py::str>(value))
    {
        return make_matrix_from_iterable(value.cast<py::iterable>());
    }

    throw Exceptions::TypeError(
        "Matrix constructor expects a Matrix, Shape, or iterable of row values.");
}

} // anonymous namespace

void bind_matrix_constructors(py::class_<Matrix>& cls)
{
    cls
        .def(py::init([]() {
            return Matrix(0, 0);
        }))
        .def(py::init([](py::object rows, py::object cols) {
            const std::size_t row_count = checked_matrix_dimension(
                binding_utils::cast_integer(rows, "Matrix row count must be an integer.", "Matrix row count is too large to fit in a signed integer."),
                "Matrix row count cannot be negative.");
            const std::size_t col_count = checked_matrix_dimension(
                binding_utils::cast_integer(cols, "Matrix column count must be an integer.", "Matrix column count is too large to fit in a signed integer."),
                "Matrix column count cannot be negative.");
            checked_matrix_elements(row_count, col_count);
            return Matrix(
                row_count,
                col_count
            );
        }), py::arg("rows"), py::arg("cols"))
        .def(py::init([](const Shape& shape) {
            return make_matrix_from_shape(shape);
        }), py::arg("shape"))
        .def(py::init<const Matrix&>(), py::arg("other"))
        .def(py::init([](py::object value) {
            return make_matrix_from_object(value);
        }), py::arg("value"))
        .def(py::init([](py::object rows, py::object cols, py::object value) {
            const std::size_t row_count = checked_matrix_dimension(
                binding_utils::cast_integer(rows, "Matrix row count must be an integer.", "Matrix row count is too large to fit in a signed integer."),
                "Matrix row count cannot be negative.");
            const std::size_t col_count = checked_matrix_dimension(
                binding_utils::cast_integer(cols, "Matrix column count must be an integer.", "Matrix column count is too large to fit in a signed integer."),
                "Matrix column count cannot be negative.");
            checked_matrix_elements(row_count, col_count);
            return Matrix(
                row_count,
                col_count,
                binding_utils::cast_scalar(
                    value,
                    "Matrix fill value must be a number.",
                    "Matrix fill value is too large to represent as a float.")
            );
        }), py::arg("rows"), py::arg("cols"), py::arg("value"));
}
