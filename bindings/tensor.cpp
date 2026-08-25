#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binding_utils/views.hpp"

#include <stratax/containers/Tensor.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Slicing.hpp>
#include <stratax/ops/Arithmetic.hpp>

#include "binding_utils/arithmetic.hpp"
#include "binding_utils/comparison.hpp"
#include "binding_utils/properties.hpp"
#include "binding_utils/reshape.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <vector>

namespace py = pybind11;
using namespace binding_utils;

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
        const long long raw = cast_integer(dim);

        if (raw < 0)
        {
            throw Exceptions::ShapeError("Tensor dimensions cannot be negative.");
        }

        const std::size_t value = static_cast<std::size_t>(raw);

        values.push_back(value);
        saw_dim = true;

        if (value != 0 &&
            elements > std::numeric_limits<std::size_t>::max() / value)
        {
            raise_overflow(
                Exceptions::OverflowError("Tensor element count overflow."));
        }

        elements *= value;

        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
        {
            raise_overflow(
                Exceptions::OverflowError("Tensor storage size overflow."));
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
        indices.push_back(
            cast_index(component));
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
        .def(
            py::init([](py::object dims) {
                if (!py::isinstance<py::iterable>(dims) ||
                    py::isinstance<py::str>(dims))
                {
                    throw Exceptions::TypeError(
                        "Tensor shape must be a Shape or iterable of dimensions.");
                }

                return Tensor(
                    make_shape_from_iterable(
                        dims.cast<py::iterable>()));
            }),
            py::arg("shape"))
        .def(
            py::init([](const Shape& shape, py::object value) {
                return Tensor(
                    shape,
                    cast_scalar(value));
            }),
            py::arg("shape"),
            py::arg("value"))
        .def(
            py::init([](py::object dims, py::object value) {
                if (!py::isinstance<py::iterable>(dims) ||
                    py::isinstance<py::str>(dims))
                {
                    throw Exceptions::TypeError(
                        "Tensor shape must be a Shape or iterable of dimensions.");
                }

                return Tensor(
                    make_shape_from_iterable(
                        dims.cast<py::iterable>()),
                    cast_scalar(value));
            }),
            py::arg("shape"),
            py::arg("value"));
}

void bind_tensor_indexing(py::class_<Tensor>& cls)
{
    cls
        .def(
            "__getitem__",
            [](py::object self, py::object index) -> py::object {
                Tensor& tensor = self.cast<Tensor&>();

                // Single slice: only currently supported for rank-1 Tensor.
                if (py::isinstance<py::slice>(index))
                {
                    if (tensor.rank() != 1)
                    {
                        throw Exceptions::RankError(
                            "The number of slices must match the tensor rank.");
                    }

                    std::vector<stratax::core::Slice> slices{
                        cast_slice(
                            index.cast<py::slice>(),
                            tensor.shape()[0])
                    };

                    auto view =
                        stratax::indexing::slice(tensor, slices);

                    return py::cast(
                        PyArrayView(
                            std::move(view),
                            self));
                }

                // Tuple indexing: a[1, 2] or a[1:3, 2:4]
                if (py::isinstance<py::tuple>(index))
                {
                    const py::tuple tuple_index =
                        index.cast<py::tuple>();

                    if (tuple_index.size() != tensor.rank())
                    {
                        throw Exceptions::RankError(
                            "The number of indices must match the tensor rank.");
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

                    // All integers -> scalar
                    if (!any_slice)
                    {
                        return py::cast(
                            tensor.at(
                                tensor_indices(tuple_index)));
                    }

                    // At least one slice -> view
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
                                cast_slice(
                                    tuple_index[dim].cast<py::slice>(),
                                    tensor.shape()[dim]));
                        }
                        else
                        {
                            ranges.push_back(
                                single_index_slice(
                                    tuple_index[dim],
                                    tensor.shape()[dim]));
                        }
                    }

                    auto view = stratax::indexing::slice(
                        tensor,
                        ranges);

                    return py::cast(
                        PyArrayView(
                            std::move(view),
                            self));
                }

                // Single integer -> flat scalar indexing.
                return py::cast(
                    tensor.at(
                        cast_index(index)));
            })
        .def(
            "__setitem__",
            [](Tensor& tensor, py::object index, double value) {
                if (py::isinstance<py::tuple>(index))
                {
                    tensor.at(
                        tensor_indices(
                            index.cast<py::tuple>())) = value;
                    return;
                }

                tensor.at(
                    cast_index(index)) = value;
            });
}

void bind_tensor(py::module_& m)
{
    py::class_<Tensor> cls(m, "Tensor");

    bind_tensor_constructors(cls);
    bind_properties(cls);
    bind_tensor_indexing(cls);
    bind_arithmetic(cls);
    bind_comparison(cls);
    bind_reshape(cls);
    bind_members(cls);
}
