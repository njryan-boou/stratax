#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binding_utils/utils.hpp"

#include <stratax/containers/Matrix.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/exceptions/ArithmeticErrors.hpp>
#include <stratax/exceptions/IndexErrors.hpp>
#include <stratax/exceptions/LayoutErrors.hpp>
#include <stratax/exceptions/TypeErrors.hpp>
#include <stratax/indexing/Slicing.hpp>
#include <stratax/io/Print.hpp>

#include "binding_utils/arithmetic.hpp"
#include "binding_utils/comparison.hpp"
#include "binding_utils/properties.hpp"
#include "binding_utils/reshape.hpp"
#include "binding_utils/views.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

// Matrix constructors

namespace py = pybind11;

using Matrix = stratax::container::Matrix<double>;

namespace
{

std::size_t checked_matrix_dimension(long long value, bool rows)
{
    if (value < 0)
    {
        throw rows
            ? Exceptions::DimensionError::negative_matrix_rows(value)
            : Exceptions::DimensionError::negative_matrix_columns(value);
    }

    return static_cast<std::size_t>(value);
}

void ensure_matrix_storage_fits(std::size_t rows, std::size_t cols)
{
    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols)
    {
        binding_utils::raise_overflow(
            Exceptions::OverflowError::matrix_size(rows, cols));
    }

    const std::size_t elements = rows * cols;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        binding_utils::raise_overflow(
            Exceptions::OverflowError::matrix_storage(
                elements, sizeof(double)));
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
            throw Exceptions::TypeError::matrix_rows();
        }

        std::vector<double> row;
        for (py::handle value : row_object.cast<py::iterable>())
        {
            row.push_back(binding_utils::cast_scalar(
                value));
        }

        if (!saw_row)
        {
            cols = row.size();
            saw_row = true;
        }
        else if (row.size() != cols)
        {
            throw Exceptions::ShapeError::ragged_matrix();
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
                binding_utils::cast_integer(rows),
                true);
            const std::size_t col_count = checked_matrix_dimension(
                binding_utils::cast_integer(cols),
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
                throw Exceptions::TypeError::matrix_constructor();
            }

            return make_matrix_from_iterable(
                value.cast<py::iterable>());
        }), py::arg("value"))
        .def(py::init([](py::object rows, py::object cols, py::object value) {
            const std::size_t row_count = checked_matrix_dimension(
                binding_utils::cast_integer(rows),
                true);
            const std::size_t col_count = checked_matrix_dimension(
                binding_utils::cast_integer(cols),
                false);
            ensure_matrix_storage_fits(row_count, col_count);
            return Matrix(
                row_count,
                col_count,
                binding_utils::cast_scalar(
                    value));
        }), py::arg("rows"), py::arg("cols"), py::arg("value"));
}

// Matrix properties

void bind_matrix_properties(py::class_<Matrix>& cls)
{
    binding_utils::bind_properties(cls);

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
        .def(
            "__getitem__",
            [](py::object self, py::object index) -> py::object {
                Matrix& matrix = self.cast<Matrix&>();

                if (!py::isinstance<py::tuple>(index))
                {
                    return py::cast(
                        matrix.at(
                            binding_utils::cast_index(index)));
                }

                const py::tuple tuple_index =
                    index.cast<py::tuple>();

                if (tuple_index.size() != 2)
                {
                    throw Exceptions::IndexError::matrix_index_rank(
                        tuple_index.size());
                }

                const bool row_slice =
                    py::isinstance<py::slice>(tuple_index[0]);
                const bool col_slice =
                    py::isinstance<py::slice>(tuple_index[1]);

                if (!row_slice && !col_slice)
                {
                    const auto row =
                        binding_utils::cast_index(tuple_index[0]);
                    const auto col =
                        binding_utils::cast_index(tuple_index[1]);

                    return py::cast(matrix.at(row, col));
                }

                const stratax::core::Slice rows = row_slice
                    ? binding_utils::cast_slice(
                        tuple_index[0].cast<py::slice>(),
                        matrix.rows())
                    : binding_utils::single_index_slice(
                        tuple_index[0],
                        matrix.rows());

                const stratax::core::Slice cols = col_slice
                    ? binding_utils::cast_slice(
                        tuple_index[1].cast<py::slice>(),
                        matrix.cols())
                    : binding_utils::single_index_slice(
                        tuple_index[1],
                        matrix.cols());

                auto view = stratax::indexing::slice(
                    matrix,
                    rows,
                    cols);

                return py::cast(
                    binding_utils::PyArrayView(
                        std::move(view),
                        self));
            })
        .def(
            "__setitem__",
            [](Matrix& matrix, py::tuple index, double value) {
                if (index.size() != 2)
                {
                    throw Exceptions::IndexError::matrix_tuple_index(
                        index.size());
                }

                matrix.at(
                    binding_utils::cast_index(index[0]),
                    binding_utils::cast_index(index[1])) = value;
            });
}

// Matrix registration

void bind_matrix(py::module_& m)
{
    py::class_<Matrix> cls(m, "Matrix");

    bind_matrix_constructors(cls);
    bind_matrix_properties(cls);
    bind_matrix_indexing(cls);
    binding_utils::bind_arithmetic(cls);
    binding_utils::bind_comparison(cls);
    binding_utils::bind_reshape(cls);
    binding_utils::bind_members(cls);
}
