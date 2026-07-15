#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/containers/Tensor.hpp>
#include <stratax/io/Print.hpp>
#include <stratax/ops/comparison.hpp>

#include <cstddef>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace
{

using Tensor = stratax::container::Tensor<double>;

} // anonymous namespace

void bind_tensor_properties(py::class_<Tensor>& cls)
{
    cls
        .def("size", &Tensor::size)
        .def("rank", &Tensor::rank)
        .def("empty", &Tensor::empty)
        .def("shape", &Tensor::shape, py::return_value_policy::reference_internal)
        .def("strides", [](const Tensor& tensor) {
            std::vector<std::size_t> values;
            const auto& strides = tensor.strides();
            values.reserve(strides.rank());
            for (std::size_t stride : strides) {
                values.push_back(stride);
            }
            return values;
        })
        .def("fill", &Tensor::fill, py::arg("value"))
        .def("tolist", [](const Tensor& tensor) {
            std::vector<double> values;
            values.reserve(tensor.size());
            for (std::size_t i = 0; i < tensor.size(); ++i) {
                values.push_back(tensor[i]);
            }
            return values;
        })
        .def("__iter__", [](const Tensor& tensor) {
            return py::make_iterator(tensor.begin(), tensor.end());
        }, py::keep_alive<0, 1>())
        .def("__eq__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs == rhs;
        })
        .def("__ne__", [](const Tensor& lhs, const Tensor& rhs) {
            return lhs != rhs;
        })
        .def("__repr__", [](const Tensor& tensor) {
            std::ostringstream os;
            os << tensor;
            return os.str();
        });
}
