#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "properties.hpp"
#include "utils.hpp"

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/io/Print.hpp>

#include <cstddef>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace
{

using BoolVector = stratax::container::Vector<bool>;
using BoolMatrix = stratax::container::Matrix<bool>;
using BoolTensor = stratax::container::Tensor<bool>;

template<typename Array>
void bind_boolean_properties(py::class_<Array>& cls)
{
    bind_properties(cls);

    cls
        .def("__len__", [](const Array& arr) { return arr.size(); })
        .def("__iter__", [](const Array& arr) {
            return py::make_iterator(arr.begin(), arr.end());
        }, py::keep_alive<0, 1>())
        .def("__repr__", [](const Array& arr) {
            std::ostringstream os;
            os << arr;
            return os.str();
        });
}

} // anonymous namespace

void bind_boolean_arrays(py::module_& m)
{
    py::class_<BoolVector> vector(m, "BoolVector");
    bind_boolean_properties(vector);
    vector.def("tolist", [](const BoolVector& arr) {
        return std::vector<bool>(arr.begin(), arr.end());
    });

    py::class_<BoolMatrix> matrix(m, "BoolMatrix");
    bind_boolean_properties(matrix);
    matrix.def("tolist", [](const BoolMatrix& arr) {
        std::vector<std::vector<bool>> values(arr.rows());
        for (std::size_t row = 0; row < arr.rows(); ++row)
        {
            values[row].reserve(arr.cols());
            for (std::size_t col = 0; col < arr.cols(); ++col)
            {
                values[row].push_back(arr(row, col));
            }
        }
        return values;
    });

    py::class_<BoolTensor> tensor(m, "BoolTensor");
    bind_boolean_properties(tensor);
    tensor.def("tolist", [](const BoolTensor& arr) {
        return binding_utils::tensor_to_list(arr);
    });
}
