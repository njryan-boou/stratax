#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.hpp"

#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/io/Print.hpp>
#include <stratax/ops/Arithmetic.hpp>

#include "comparison.hpp"
#include "arithmetic.hpp"
#include "reshape.hpp"
#include "properties.hpp"

#include <cstddef>
#include <limits>
#include <sstream>
#include <vector>

// =============================================================================
// Tensor constructors
// =============================================================================

namespace py = pybind11;

using Tensor = stratax::container::Tensor<double>;

namespace
{

using Shape = stratax::core::Shape;

void require_allocatable_tensor_elements(std::size_t elements)
{
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        binding_utils::raise_overflow("Tensor storage size overflow.");
    }
} // anonymous namespace

std::size_t checked_tensor_elements(const Shape& shape)
{
    try
    {
        const std::size_t elements = shape.elements();
        require_allocatable_tensor_elements(elements);
        return elements;
    }
    catch (const Exceptions::DimensionError& e)
    {
        binding_utils::raise_overflow(e.what());
    }
}

Shape make_shape(const std::vector<long long>& dims)
{
    std::vector<std::size_t> values;
    values.reserve(dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
        values.push_back(stratax::core::validation::nonnegative_shape_dimension(
            dims[i],
            "Tensor shape dimension cannot be negative."));
    }

    Shape shape(values);
    checked_tensor_elements(shape);
    return shape;
}

Shape make_shape(py::iterable dims)
{
    std::vector<long long> values;

    for (py::handle dim : dims) {
        values.push_back(binding_utils::cast_integer(
            dim,
            "Tensor shape dimensions must be integers.",
            "Tensor shape dimension is too large to fit in a signed integer."));
    }

    return make_shape(values);
}

Shape make_shape_from_object(py::object dims)
{
    if (py::isinstance<py::iterable>(dims) && !py::isinstance<py::str>(dims))
    {
        return make_shape(dims.cast<py::iterable>());
    }

    throw Exceptions::TypeError("Tensor shape must be a Shape or iterable of dimensions.");
}

Tensor make_tensor_from_shape(const Shape& shape)
{
    checked_tensor_elements(shape);
    return Tensor(shape);
}

Tensor make_tensor_from_shape(const Shape& shape, double value)
{
    checked_tensor_elements(shape);
    return Tensor(shape, value);
}

} // anonymous namespace

void bind_tensor_constructors(py::class_<Tensor>& cls)
{
    cls
        .def(py::init<>())
        .def(py::init([](const Shape& shape) {
            return make_tensor_from_shape(shape);
        }), py::arg("shape"))
        .def(py::init([](const Shape& shape, py::object value) {
            return make_tensor_from_shape(
                shape,
                binding_utils::cast_scalar(
                    value,
                    "Tensor fill value must be a number.",
                    "Tensor fill value is too large to represent as a float."));
        }), py::arg("shape"), py::arg("value"))
        .def(py::init<const Tensor&>(), py::arg("other"))
        .def(py::init([](py::object dims) {
            return Tensor(make_shape_from_object(dims));
        }), py::arg("shape"))
        .def(py::init([](py::object dims, py::object value) {
            return Tensor(
                make_shape_from_object(dims),
                binding_utils::cast_scalar(
                    value,
                    "Tensor fill value must be a number.",
                    "Tensor fill value is too large to represent as a float."));
        }), py::arg("shape"), py::arg("value"));
}


// =============================================================================
// Tensor properties
// =============================================================================


namespace
{
} // anonymous namespace

void bind_tensor_properties(py::class_<Tensor>& cls)
{
    bind_properties(cls);

    cls
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
        .def("__repr__", [](const Tensor& tensor) {
            std::ostringstream os;
            os << tensor;
            return os.str();
        });
}

// =============================================================================
// Tensor indexing
// =============================================================================


namespace
{
std::size_t tensor_offset(const Tensor& tensor, py::tuple index)
{
    const auto& shape = tensor.shape();
    if (index.size() != shape.rank()) {
        throw Exceptions::IndexError("Tensor index rank must match tensor rank.");
    }

    std::size_t offset = 0;
    std::size_t stride = 1;

    for (std::size_t reverse_dim = shape.rank(); reverse_dim > 0; --reverse_dim) {
        const std::size_t dim = reverse_dim - 1;
        const std::size_t component = binding_utils::resolve_index(
            index[dim],
            shape(dim),
            "Tensor index components must be integers.",
            "Tensor index component is too large to fit in a signed integer.",
            "Tensor index component is out of bounds.");

        try
        {
            const std::size_t component_offset =
                stratax::core::validation::checked_multiply(
                    component,
                    stride,
                    "Tensor index offset multiplication overflow.");

            offset = stratax::core::validation::checked_add(
                offset,
                component_offset,
                "Tensor index offset addition overflow.");

            stride = stratax::core::validation::checked_multiply(
                stride,
                shape(dim),
                "Tensor index stride overflow.");
        }
        catch (const Exceptions::DimensionError& e)
        {
            binding_utils::raise_overflow(e.what());
        }
    }

    return offset;
}

Tensor tensor_slice_runtime(const Tensor& tensor, const std::vector<binding_utils::ResolvedSlice>& ranges)
{
    stratax::core::validation::require_rank(
        ranges.size(),
        tensor.rank(),
        "Slice rank must match tensor rank.");

    std::vector<std::size_t> out_dims;
    out_dims.reserve(ranges.size());
    for (const auto& range : ranges)
    {
        out_dims.push_back(static_cast<std::size_t>(range.length));
    }

    const stratax::core::Shape result_shape(out_dims);
    Tensor result(result_shape);
    const stratax::core::Strides result_strides(result_shape);
    const auto& tensor_strides = tensor.strides();

    if (result.empty())
    {
        return result;
    }

    for (std::size_t flat = 0; flat < result.size(); ++flat)
    {
        std::size_t remainder = flat;
        std::size_t source_offset = 0;

        for (std::size_t dim = 0; dim < ranges.size(); ++dim)
        {
            const std::size_t index = remainder / result_strides(dim);
            remainder %= result_strides(dim);

            const py::ssize_t source_index = ranges[dim].start +
                static_cast<py::ssize_t>(index) * ranges[dim].step;

            const std::size_t term = stratax::core::validation::checked_multiply(
                static_cast<std::size_t>(source_index),
                tensor_strides(dim),
                "Tensor slice offset overflow.");

            source_offset = stratax::core::validation::checked_add(
                source_offset,
                term,
                "Tensor slice offset overflow.");
        }

        result[flat] = tensor[source_offset];
    }

    return result;
}

} // anonymous namespace

void bind_tensor_indexing(py::class_<Tensor>& cls)
{
    cls
        .def("__len__", &Tensor::size)
        .def("__getitem__", [](const Tensor& tensor, py::object index) -> py::object {
            if (py::isinstance<py::slice>(index)) {
                if (tensor.rank() != 1)
                {
                    throw Exceptions::IndexError("Tensor slice rank must match tensor rank.");
                }

                const auto range = binding_utils::resolve_slice(
                    index.cast<py::slice>(),
                    tensor.shape()(0),
                    "Tensor slice step cannot be zero.");
                return py::cast(tensor_slice_runtime(tensor, std::vector<binding_utils::ResolvedSlice>{range}));
            }

            if (py::isinstance<py::tuple>(index)) {
                const py::tuple tuple_index = index.cast<py::tuple>();

                if (tuple_index.size() != tensor.rank())
                {
                    throw Exceptions::IndexError("Tensor index rank must match tensor rank.");
                }

                bool any_slice = false;
                for (std::size_t dim = 0; dim < tuple_index.size(); ++dim)
                {
                    if (py::isinstance<py::slice>(tuple_index[dim]))
                    {
                        any_slice = true;
                        break;
                    }
                }

                if (!any_slice)
                {
					return py::cast(tensor[tensor_offset(tensor, tuple_index)]);
                }

                std::vector<binding_utils::ResolvedSlice> ranges;
                ranges.reserve(tuple_index.size());

                for (std::size_t dim = 0; dim < tuple_index.size(); ++dim)
                {
                    if (py::isinstance<py::slice>(tuple_index[dim]))
                    {
                        ranges.push_back(binding_utils::resolve_slice(
                            tuple_index[dim].cast<py::slice>(),
                            tensor.shape()(dim),
                            "Tensor slice step cannot be zero."));
                    }
                    else
                    {
                        ranges.push_back(binding_utils::single_index_slice(
                            tuple_index[dim],
                            tensor.shape()(dim),
                            "Tensor index components must be integers.",
                            "Tensor index component is too large to fit in a signed integer.",
                            "Tensor index component is out of bounds."));
                    }
                }

                return py::cast(tensor_slice_runtime(tensor, ranges));
            }

			return py::cast(tensor[
                binding_utils::resolve_index(
                    index,
                    tensor.size(),
                    "Tensor index must be an integer.",
                    "Tensor index is too large to fit in a signed integer.",
                    "Tensor index is out of bounds.")
			]);
        })
        .def("__setitem__", [](Tensor& tensor, py::object index, double value) {
            if (py::isinstance<py::tuple>(index)) {
				tensor[tensor_offset(tensor, index.cast<py::tuple>())] = value;
                return;
            }

			tensor[
                binding_utils::resolve_index(
                    index,
                    tensor.size(),
                    "Tensor index must be an integer.",
                    "Tensor index is too large to fit in a signed integer.",
                    "Tensor index is out of bounds.")
			] = value;
        });
}


// =============================================================================
// Tensor registration
// =============================================================================

void bind_tensor(py::module_& m)
{
    py::class_<Tensor> cls(m, "Tensor");

    bind_tensor_constructors(cls);
    bind_tensor_properties(cls);
    bind_tensor_indexing(cls);
    bind_arithmetic(cls);
    bind_comparison(cls);
    bind_reshape(cls);
}
