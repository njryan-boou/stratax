#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/core/containers/Matrix.hpp>
#include <stratax/core/io/Print.hpp>

#include <cstddef>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace
{

using Matrix = stratax::container::Matrix<double>;

}

void bind_matrix_properties(py::class_<Matrix>& cls)
{
    cls
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
        .def("__repr__", [](const Matrix& matrix) {
            std::ostringstream os;
            os << matrix;
            return os.str();
        });
}
