#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binding_utils/arithmetic.hpp"
#include "binding_utils/comparison.hpp"
#include "binding_utils/properties.hpp"
#include "binding_utils/reshape.hpp"
#include "binding_utils/views.hpp"

#include <stratax/containers/Vector.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Slicing.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace binding_utils;

namespace
{

using Vector = stratax::container::Vector<double>;

std::size_t checked_vector_size(long long size)
{
    if (size < 0)
    {
        throw Exceptions::DimensionError("Vector size cannot be negative.");
    }

    return static_cast<std::size_t>(size);
}

Vector make_vector_from_iterable(py::iterable values)
{
    std::vector<double> parsed;

    for (py::handle value : values)
    {
        parsed.push_back(
            cast_scalar(value));
    }

    Vector vector(parsed.size());
    std::copy(parsed.begin(), parsed.end(), vector.begin());

    return vector;
}

Vector make_vector_from_object(py::object value)
{
    if (py::isinstance<py::int_>(value))
    {
        return Vector(
            checked_vector_size(
                cast_integer(value)));
    }

    if (py::isinstance<py::iterable>(value) &&
        !py::isinstance<py::str>(value))
    {
        return make_vector_from_iterable(
            value.cast<py::iterable>());
    }

    throw Exceptions::TypeError(
        "Vector constructor expects a Vector, size, or iterable of numbers.");
}

} // anonymous namespace

void bind_vector_constructors(py::class_<Vector>& cls)
{
    cls
        .def(py::init<>())
        .def(py::init<const Vector&>(), py::arg("other"))
        .def(
            py::init([](py::object value) {
                return make_vector_from_object(value);
            }),
            py::arg("value"))
        .def(
            py::init([](py::object size, py::object value) {
                const auto checked_size = checked_vector_size(
                    cast_integer(size));

                const auto checked_value =
                    cast_scalar(value);

                return Vector(checked_size, checked_value);
            }),
            py::arg("size"),
            py::arg("value"));
}

void bind_vector_indexing(py::class_<Vector>& cls)
{
    cls
        .def(
            "__getitem__",
            [](py::object self, py::object index) -> py::object {
                Vector& vector = self.cast<Vector&>();

                if (py::isinstance<py::slice>(index))
                {
                    const auto slice = cast_slice(
                        index.cast<py::slice>(),
                        vector.size());

                    auto view = stratax::indexing::slice(
                        vector,
                        slice);

                    return py::cast(
                        PyArrayView(
                            std::move(view),
                            self));
                }

                return py::cast(
                    vector.at(cast_index(index)));
            })
        .def(
            "__setitem__",
            [](Vector& vector, py::object index, py::object value) {
                vector.at(cast_index(index)) = cast_scalar(value);
            });
}

void bind_vector(py::module_& m)
{
    py::class_<Vector> cls(m, "Vector");

    bind_vector_constructors(cls);
    bind_properties(cls);
    bind_vector_indexing(cls);
    bind_arithmetic(cls);
    bind_comparison(cls);
    bind_reshape(cls);
    bind_members(cls);
}
