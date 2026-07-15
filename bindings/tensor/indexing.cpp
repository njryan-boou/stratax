#include <pybind11/pybind11.h>

#include <stratax/containers/Tensor.hpp>

namespace py = pybind11;

namespace
{

using Tensor = stratax::container::Tensor<double>;

} // anonymous namespace

void bind_tensor_indexing(py::class_<Tensor>& cls)
{
    cls
        .def("__len__", &Tensor::size);
}
