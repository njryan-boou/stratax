#include <pybind11/pybind11.h>

#include <stratax/core/containers/Tensor.hpp>

namespace py = pybind11;

void bind_tensor_constructors(py::class_<stratax::container::Tensor<double>>&);
void bind_tensor_indexing(py::class_<stratax::container::Tensor<double>>&);
void bind_tensor_arithmetic(py::class_<stratax::container::Tensor<double>>&);
void bind_tensor_comparison(py::class_<stratax::container::Tensor<double>>&);
void bind_tensor_reshape(py::class_<stratax::container::Tensor<double>>&);
void bind_tensor_properties(py::class_<stratax::container::Tensor<double>>&);

void bind_tensor(py::module_& m)
{
    using Tensor = stratax::container::Tensor<double>;

    py::class_<Tensor> cls(m, "Tensor");

    bind_tensor_constructors(cls);
    bind_tensor_properties(cls);
    bind_tensor_indexing(cls);
    bind_tensor_arithmetic(cls);
    bind_tensor_comparison(cls);
    bind_tensor_reshape(cls);
}
