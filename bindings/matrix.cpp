#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/containers/Matrix.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Validation.hpp>
#include <stratax/io/Print.hpp>
#include <stratax/ops/Arithmetic.hpp>
#include <stratax/ops/Comparison.hpp>

#include <cstddef>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{

using Matrix = stratax::container::Matrix<double>;
using Shape = stratax::core::Shape;

[[noreturn]] void raise_overflow(const char* message)
{
    PyErr_SetString(PyExc_OverflowError, message);
    throw py::error_already_set();
}

long long cast_integer(py::handle value, const char* type_message, const char* overflow_message)
{
    if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value))
    {
        throw Exceptions::TypeError(type_message);
    }

    PyErr_Clear();
    const long long result = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred())
    {
        PyErr_Clear();
        raise_overflow(overflow_message);
    }

    return result;
}

void require_allocatable_matrix_elements(std::size_t elements)
{
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        raise_overflow("Matrix storage size overflow.");
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
        raise_overflow(e.what());
    }
}

std::size_t checked_matrix_dimension(long long value, const char* message)
{
    return stratax::core::validation::nonnegative_size(value, message);
}

double cast_scalar(py::handle value, const char* type_message, const char* overflow_message)
{
    if (py::isinstance<py::bool_>(value)
        || !(py::isinstance<py::int_>(value) || py::isinstance<py::float_>(value)))
    {
        throw Exceptions::TypeError(type_message);
    }

    PyErr_Clear();
    const double result = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred())
    {
        PyErr_Clear();
        raise_overflow(overflow_message);
    }

    if (!std::isfinite(result))
    {
        raise_overflow(overflow_message);
    }

    return result;
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
            row.push_back(cast_scalar(
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

void bind_matrix(py::module_& m)
{
    using Matrix = stratax::container::Matrix<double>;
    using Shape = stratax::core::Shape;

    py::class_<Matrix>(m, "Matrix")
        // Constructors
        .def(py::init([]() {
            return Matrix(0, 0);
        }))
        .def(py::init([](py::object rows, py::object cols) {
            const std::size_t row_count = checked_matrix_dimension(
                cast_integer(rows, "Matrix row count must be an integer.", "Matrix row count is too large to fit in a signed integer."),
                "Matrix row count cannot be negative.");
            const std::size_t col_count = checked_matrix_dimension(
                cast_integer(cols, "Matrix column count must be an integer.", "Matrix column count is too large to fit in a signed integer."),
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
                cast_integer(rows, "Matrix row count must be an integer.", "Matrix row count is too large to fit in a signed integer."),
                "Matrix row count cannot be negative.");
            const std::size_t col_count = checked_matrix_dimension(
                cast_integer(cols, "Matrix column count must be an integer.", "Matrix column count is too large to fit in a signed integer."),
                "Matrix column count cannot be negative.");
            checked_matrix_elements(row_count, col_count);
            return Matrix(
                row_count,
                col_count,
                cast_scalar(
                    value,
                    "Matrix fill value must be a number.",
                    "Matrix fill value is too large to represent as a float.")
            );
        }), py::arg("rows"), py::arg("cols"), py::arg("value"))

        // Methods
        .def("size", &Matrix::size)
        .def("rank", &Matrix::rank)
        .def("empty", &Matrix::empty)
        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)
        .def("shape", &Matrix::shape, py::return_value_policy::reference_internal)
        .def("strides", [](const Matrix& matrix) {
            std::vector<std::size_t> values;
            const auto& strides = matrix.strides();
            values.reserve(strides.rank());
            for (std::size_t stride : strides) {
                values.push_back(stride);
            }
            return values;
        })
        .def("fill", &Matrix::fill, py::arg("value"))
        .def("tolist", [](const Matrix& matrix) {
            std::vector<std::vector<double>> values(matrix.rows());
            for (std::size_t row = 0; row < matrix.rows(); ++row) {
                values[row].reserve(matrix.cols());
                for (std::size_t col = 0; col < matrix.cols(); ++col) {
                    values[row].push_back(matrix(row, col));
                }
            }
            return values;
        })

        // Python special methods
        .def("__len__", &Matrix::rows)
        .def("__getitem__", [](const Matrix& matrix, py::tuple index) {
            if (index.size() != 2) {
                throw Exceptions::IndexError("Matrix index must be a (row, column) tuple.");
            }

            return matrix.at(
                stratax::core::validation::nonnegative_index(
                    cast_integer(index[0], "Matrix row index must be an integer.", "Matrix row index is too large to fit in a signed integer."),
                    "Matrix row index cannot be negative."),
                stratax::core::validation::nonnegative_index(
                    cast_integer(index[1], "Matrix column index must be an integer.", "Matrix column index is too large to fit in a signed integer."),
                    "Matrix column index cannot be negative.")
            );
        })
        .def("__setitem__", [](Matrix& matrix, py::tuple index, double value) {
            if (index.size() != 2) {
                throw Exceptions::IndexError("Matrix index must be a (row, column) tuple.");
            }

            matrix.at(
                stratax::core::validation::nonnegative_index(
                    cast_integer(index[0], "Matrix row index must be an integer.", "Matrix row index is too large to fit in a signed integer."),
                    "Matrix row index cannot be negative."),
                stratax::core::validation::nonnegative_index(
                    cast_integer(index[1], "Matrix column index must be an integer.", "Matrix column index is too large to fit in a signed integer."),
                    "Matrix column index cannot be negative.")
            ) = value;
        })
        .def("__iter__", [](const Matrix& matrix) {
            return py::make_iterator(matrix.begin(), matrix.end());
        }, py::keep_alive<0, 1>())
        .def("__eq__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs == rhs;
        })
        .def("__ne__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs != rhs;
        })
        .def("__add__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs + rhs;
        })
        .def("__add__", [](const Matrix& lhs, double rhs) {
            return lhs + rhs;
        })
        .def("__radd__", [](const Matrix& rhs, double lhs) {
            return lhs + rhs;
        })
        .def("__sub__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs - rhs;
        })
        .def("__sub__", [](const Matrix& lhs, double rhs) {
            return lhs - rhs;
        })
        .def("__rsub__", [](const Matrix& rhs, double lhs) {
            return lhs - rhs;
        })
        .def("__mul__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs * rhs;
        })
        .def("__mul__", [](const Matrix& lhs, double rhs) {
            return lhs * rhs;
        })
        .def("__rmul__", [](const Matrix& rhs, double lhs) {
            return lhs * rhs;
        })
        .def("__truediv__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs / rhs;
        })
        .def("__truediv__", [](const Matrix& lhs, double rhs) {
            return lhs / rhs;
        })
        .def("__rtruediv__", [](const Matrix& rhs, double lhs) {
            return lhs / rhs;
        })
        .def("__iadd__", [](Matrix& lhs, const Matrix& rhs) -> Matrix& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__iadd__", [](Matrix& lhs, double rhs) -> Matrix& {
            lhs += rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Matrix& lhs, const Matrix& rhs) -> Matrix& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Matrix& lhs, double rhs) -> Matrix& {
            lhs -= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Matrix& lhs, const Matrix& rhs) -> Matrix& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Matrix& lhs, double rhs) -> Matrix& {
            lhs *= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Matrix& lhs, const Matrix& rhs) -> Matrix& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Matrix& lhs, double rhs) -> Matrix& {
            lhs /= rhs;
            return lhs;
        }, py::return_value_policy::reference_internal)
        .def("__pos__", [](const Matrix& matrix) {
            return +matrix;
        })
        .def("__neg__", [](const Matrix& matrix) {
            return -matrix;
        })
        .def("__repr__", [](const Matrix& matrix) {
            std::ostringstream os;
            os << matrix;
            return os.str();
        });
}
