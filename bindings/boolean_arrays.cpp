#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binding_utils/properties.hpp"

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

#include <cstddef>

namespace py = pybind11;

namespace
{

using BoolVector = stratax::container::Vector<bool>;
using BoolMatrix = stratax::container::Matrix<bool>;
using BoolTensor = stratax::container::Tensor<bool>;

template<typename Array>
void bind_boolean_properties(py::class_<Array>& cls)
{
    binding_utils::bind_properties(cls);

    cls
        .def("__len__", [](const Array& arr) { return arr.size(); })
        .def("__iter__", [](const Array& arr) {
            return py::make_iterator(arr.begin(), arr.end());
        }, py::keep_alive<0, 1>());
}

} // anonymous namespace

void bind_boolean_arrays(py::module_& m)
{
    py::class_<BoolVector> vector(m, "BoolVector");
    bind_boolean_properties(vector);

    py::class_<BoolMatrix> matrix(m, "BoolMatrix");
    bind_boolean_properties(matrix);

    py::class_<BoolTensor> tensor(m, "BoolTensor");
    bind_boolean_properties(tensor);
}
