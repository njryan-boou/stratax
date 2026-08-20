#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binding_utils/utils.hpp"

#include <stratax/containers/Tensor.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Slicing.hpp>
#include <stratax/io/Print.hpp>
#include <stratax/ops/Arithmetic.hpp>

#include "binding_utils/arithmetic.hpp"
#include "binding_utils/comparison.hpp"
#include "binding_utils/properties.hpp"
#include "binding_utils/reshape.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

namespace py = pybind11;

using Tensor = stratax::container::Tensor<double>;

namespace
{

using Shape = stratax::core::Shape;

Shape make_shape_from_iterable(py::iterable dims)
{
    std::vector<std::size_t> values;
    std::size_t elements = 1;
    bool saw_dim = false;

    for (py::handle dim : dims)
    {
        const long long raw = binding_utils::cast_integer(
            dim,
            "Tensor shape dimensions must be integers.",
            "Tensor shape dimension is too large to fit in a signed integer.");

        const std::size_t value = stratax::core::validation::nonnegative_shape_dimension(
            raw,
            "Tensor shape dimensions cannot be negative.");

        values.push_back(value);
        saw_dim = true;

        try
        {
            elements = stratax::core::validation::checked_multiply(
                elements,
                value,
                "Tensor shape element count overflow.");
        }
        catch (const Exceptions::DimensionError& e)
        {
            binding_utils::raise_overflow(e.what());
        }

        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
        {
            binding_utils::raise_overflow("Tensor storage size overflow.");
        }
    }

    if (!saw_dim)
    {
        return Shape();
    }

    return Shape(values);
}

std::vector<std::ptrdiff_t> tensor_indices(py::tuple index)
{
    std::vector<std::ptrdiff_t> indices;
    indices.reserve(index.size());

    for (py::handle component : index)
    {
        indices.push_back(binding_utils::cast_index(
            component,
            "Tensor index components must be integers.",
            "Tensor index component is too large to fit in a signed integer."));
    }

    return indices;
}

} // anonymous namespace

void bind_tensor_constructors(py::class_<Tensor>& cls)
{
    cls
        .def(py::init<>())
        .def(py::init<const Shape&>(), py::arg("shape"))
        .def(py::init<const Tensor&>(), py::arg("other"))
        .def(py::init([](py::object dims) {
            if (!py::isinstance<py::iterable>(dims) || py::isinstance<py::str>(dims))
            {
                throw Exceptions::TypeError(
                    "Tensor shape must be a Shape or iterable of dimensions.");
            }

            return Tensor(
                make_shape_from_iterable(dims.cast<py::iterable>()));
        }), py::arg("shape"))
        .def(py::init([](const Shape& shape, py::object value) {
            return Tensor(
                shape,
                binding_utils::cast_scalar(
                    value,
                    "Tensor fill value must be a number.",
                    "Tensor fill value is too large to represent as a float."));
        }), py::arg("shape"), py::arg("value"))
        .def(py::init([](py::object dims, py::object value) {
            if (!py::isinstance<py::iterable>(dims) || py::isinstance<py::str>(dims))
            {
                throw Exceptions::TypeError(
                    "Tensor shape must be a Shape or iterable of dimensions.");
            }

            return Tensor(
                make_shape_from_iterable(dims.cast<py::iterable>()),
                binding_utils::cast_scalar(
                    value,
                    "Tensor fill value must be a number.",
                    "Tensor fill value is too large to represent as a float."));
        }), py::arg("shape"), py::arg("value"));
}

void bind_tensor_properties(py::class_<Tensor>& cls)
{
    binding_utils::bind_properties(cls);

    cls
        .def("tolist", [](const Tensor& tensor) {
            return binding_utils::tensor_to_list(tensor);
        })
        .def("__iter__", [](const Tensor& tensor) {
            return py::make_iterator(tensor.begin(), tensor.end());
        }, py::keep_alive<0, 1>())
        .def("__repr__", [](const Tensor& tensor) {
            std::ostringstream os;
            os << tensor;
            return os.str();
        });
}

void bind_tensor_indexing(py::class_<Tensor>& cls)
{
    cls
        .def("__len__", [](const Tensor& tensor) { return tensor.size(); })
        .def(
    "__getitem__",
    [](py::object self, py::object index) -> py::object
    {
        Tensor& tensor = self.cast<Tensor&>();

        if (py::isinstance<py::slice>(index))
        {
            if (tensor.rank() != 1)
            {
                throw Exceptions::IndexError(
                    "Tensor slice rank must match tensor rank.");
            }

            std::vector<stratax::core::Slice> slices{
                binding_utils::cast_slice(
                    index.cast<py::slice>(),
                    tensor.shape()[0])
            };

            auto view =
                stratax::indexing::slice(tensor, slices);

            return py::cast(
                std::move(view),
                py::return_value_policy::move,
                self);
        }

        if (py::isinstance<py::tuple>(index))
        {
            const py::tuple tuple_index =
                index.cast<py::tuple>();

            if (tuple_index.size() != tensor.rank())
            {
                throw Exceptions::IndexError(
                    "Tensor index rank must match tensor rank.");
            }

            bool any_slice = false;

            for (std::size_t dim = 0;
                 dim < tuple_index.size();
                 ++dim)
            {
                if (py::isinstance<py::slice>(
                        tuple_index[dim]))
                {
                    any_slice = true;
                    break;
                }
            }

            if (!any_slice)
            {
                return py::cast(
                    tensor.at(
                        tensor_indices(tuple_index)));
            }

            std::vector<stratax::core::Slice> ranges;
            ranges.reserve(tuple_index.size());

            for (std::size_t dim = 0;
                 dim < tuple_index.size();
                 ++dim)
            {
                if (py::isinstance<py::slice>(
                        tuple_index[dim]))
                {
                    ranges.push_back(
                        binding_utils::cast_slice(
                            tuple_index[dim]
                                .cast<py::slice>(),
                            tensor.shape()[dim]));
                }
                else
                {
                    ranges.push_back(
                        binding_utils::single_index_slice(
                            tuple_index[dim],
                            tensor.shape()[dim],
                            "Tensor index components must be integers.",
                            "Tensor index component is too large to fit in a signed integer."));
                }
            }

            auto view =
                stratax::indexing::slice(
                    tensor,
                    ranges);

            return py::cast(
                std::move(view),
                py::return_value_policy::move,
                self);
        }

        return py::cast(
            tensor.at(
                binding_utils::cast_index(
                    index,
                    "Tensor index must be an integer.",
                    "Tensor index is too large to fit in a signed integer.")));
    })
        .def("__setitem__", [](Tensor& tensor, py::object index, double value) {
            if (py::isinstance<py::tuple>(index))
            {
                tensor.at(tensor_indices(index.cast<py::tuple>())) = value;
                return;
            }

            tensor.at(binding_utils::cast_index(
                index,
                "Tensor index must be an integer.",
                "Tensor index is too large to fit in a signed integer.")) = value;
        });
}

void bind_tensor(py::module_& m)
{
    py::class_<Tensor> cls(m, "Tensor");

    bind_tensor_constructors(cls);
    bind_tensor_properties(cls);
    bind_tensor_indexing(cls);
    binding_utils::bind_arithmetic(cls);
    binding_utils::bind_comparison(cls);
    binding_utils::bind_reshape(cls);
}
