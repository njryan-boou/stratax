#include <pybind11/pybind11.h>

#include <stratax/core/containers/Matrix.hpp>
#include <stratax/core/ops/Comparison.hpp>

namespace py = pybind11;

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
