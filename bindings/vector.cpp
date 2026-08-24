#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binding_utils/arithmetic.hpp"
#include "binding_utils/comparison.hpp"
#include "binding_utils/properties.hpp"
#include "binding_utils/reshape.hpp"
#include "binding_utils/utils.hpp"
#include "binding_utils/views.hpp"

#include <stratax/containers/Vector.hpp>
#include <stratax/exceptions/LayoutErrors.hpp>
#include <stratax/exceptions/TypeErrors.hpp>
#include <stratax/indexing/Slicing.hpp>
#include <stratax/io/Print.hpp>

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

using Vector = stratax::container::Vector<double>;

std::size_t checked_vector_size(long long size)
{
    if (size < 0)
    {
        throw Exceptions::DimensionError::negative_vector_size(size);
    }

    return static_cast<std::size_t>(size);
}

Vector make_vector_from_iterable(py::iterable values)
{
    std::vector<double> parsed;

    for (py::handle value : values)
    {
        parsed.push_back(
            binding_utils::cast_scalar(value));
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
                binding_utils::cast_integer(value)));
    }

    if (py::isinstance<py::iterable>(value) &&
        !py::isinstance<py::str>(value))
    {
        return make_vector_from_iterable(
            value.cast<py::iterable>());
    }

    throw Exceptions::TypeError::vector_constructor();
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
                    binding_utils::cast_integer(size));

                const auto checked_value =
                    binding_utils::cast_scalar(value);

                return Vector(checked_size, checked_value);
            }),
            py::arg("size"),
            py::arg("value"));
}

void bind_vector_properties(py::class_<Vector>& cls)
{
    binding_utils::bind_properties(cls);

    cls
        .def("tolist", [](const Vector& vector) {
            return std::vector<double>(
                vector.begin(),
                vector.end());
        })
        .def("__repr__", [](const Vector& vector) {
            std::ostringstream os;
            os << vector;
            return os.str();
        });
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
                    const auto slice = binding_utils::cast_slice(
                        index.cast<py::slice>(),
                        vector.size());

                    auto view = stratax::indexing::slice(
                        vector,
                        slice);

                    return py::cast(
                        binding_utils::PyArrayView(
                            std::move(view),
                            self));
                }

                return py::cast(
                    vector.at(
                        binding_utils::cast_index(index)));
            })
        .def(
            "__setitem__",
            [](Vector& vector, py::object index, py::object value) {
                vector.at(
                    binding_utils::cast_index(index)) =
                    binding_utils::cast_scalar(value);
            });
}

void bind_vector(py::module_& m)
{
    py::class_<Vector> cls(m, "Vector");

    bind_vector_constructors(cls);
    bind_vector_properties(cls);
    bind_vector_indexing(cls);
    binding_utils::bind_arithmetic(cls);
    binding_utils::bind_comparison(cls);
    binding_utils::bind_reshape(cls);
    binding_utils::bind_members(cls);
}
