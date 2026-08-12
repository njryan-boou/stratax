#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.hpp"

#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/algorithms/Reshape.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/io/Print.hpp>
#include <stratax/ops/Arithmetic.hpp>
#include <stratax/ops/Comparison.hpp>

#include <cstddef>
#include <sstream>
#include <vector>
#include <utility>
#include <limits>

// =============================================================================
// Matrix constructors
// =============================================================================

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
} // anonymous namespace

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
} // anonymous namespace

std::size_t checked_matrix_dimension(long long value, const char* message)
{
    return stratax::core::validation::nonnegative_size(value, message);
} // anonymous namespace

Matrix make_matrix_from_shape(const Shape& shape)
{
    stratax::core::validation::require_rank(shape, 2, "Matrix shape must have rank 2.");
    checked_matrix_elements(shape(0), shape(1));
    return Matrix(shape);
} // anonymous namespace

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
} // anonymous namespace

Matrix make_matrix_from_object(py::object value)
{
    if (py::isinstance<py::iterable>(value) && !py::isinstance<py::str>(value))
    {
        return make_matrix_from_iterable(value.cast<py::iterable>());
    }

    throw Exceptions::TypeError(
        "Matrix constructor expects a Matrix, Shape, or iterable of row values.");
} // anonymous namespace

}

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


// =============================================================================
// Matrix properties
// =============================================================================


namespace
{

using Matrix = stratax::container::Matrix<double>;

}

void bind_matrix_properties(py::class_<Matrix>& cls)
{
    cls
        .def_property_readonly("size", &Matrix::size)
        .def_property_readonly("rank", &Matrix::rank)
        .def_property_readonly("empty", &Matrix::empty)
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def_property_readonly("shape", &Matrix::shape, py::return_value_policy::reference_internal)
        .def_property_readonly("strides", [](const Matrix& matrix) {
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
        .def("__repr__", [](const Matrix& matrix) {
            std::ostringstream os;
            os << matrix;
            return os.str();
        });
}

// =============================================================================
// Matrix indexing
// =============================================================================


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

}

void bind_matrix_indexing(py::class_<Matrix>& cls)
{
    cls
        .def("__len__", &Matrix::rows)
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
            if (tuple_index.size() != 2)
            {
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
        })
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

// =============================================================================
// Matrix arithmetic
// =============================================================================


namespace
{

using Matrix = stratax::container::Matrix<double>;

}

void bind_matrix_arithmetic(py::class_<Matrix>& cls)
{
    cls
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
        });
}

// =============================================================================
// Matrix comparison
// =============================================================================


namespace
{

using Matrix = stratax::container::Matrix<double>;

}

void bind_matrix_comparison(py::class_<Matrix>& cls)
{
    cls
        .def("__eq__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs == rhs;
        })
        .def("__ne__", [](const Matrix& lhs, const Matrix& rhs) {
            return lhs != rhs;
        });
}

// =============================================================================
// Matrix reshape
// =============================================================================


namespace
{

using Matrix = stratax::container::Matrix<double>;

}

void bind_matrix_reshape(py::class_<Matrix>& cls)
{
    cls
        .def(
            "reshape",
            [](const Matrix& self,
               const stratax::core::Shape& shape)
            {
                return reshape(self, shape);
            },
            py::arg("shape"),
            "Return a reshaped tensor.")
        .def(
            "reshape",
            [](const Matrix& self,
               const std::vector<std::size_t>& dims)
            {
                return reshape(self, stratax::core::Shape(dims));
            },
            py::arg("shape"),
            "Return a reshaped tensor.")
        .def(
            "flatten",
            [](const Matrix& self)
            {
                return flatten(self);
            },
            "Return a flattened vector.");
}

// =============================================================================
// Matrix registration
// =============================================================================

void bind_matrix(py::module_& m)
{
    using Matrix = stratax::container::Matrix<double>;

    py::class_<Matrix> cls(m, "Matrix");

    bind_matrix_constructors(cls);
    bind_matrix_properties(cls);
    bind_matrix_indexing(cls);
    bind_matrix_arithmetic(cls);
    bind_matrix_comparison(cls);
    bind_matrix_reshape(cls);
}
