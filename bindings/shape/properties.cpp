#include <pybind11/pybind11.h>

#include "../common/utils.hpp"

#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Validation.hpp>

#include <sstream>

namespace py = pybind11;

namespace
{

using Shape = stratax::core::Shape;

} // anonymous namespace

void bind_shape_properties(py::class_<Shape>& cls)
{
    cls
        .def("rank", &Shape::rank)
        .def("elements", [](const Shape& shape)
        {
            try
            {
                return shape.elements();
            }
            catch (const Exceptions::DimensionError& e)
            {
                binding_utils::raise_overflow(e.what());
            }
        })
        .def("empty", &Shape::empty)
        .def("__len__", &Shape::rank)
        .def("__getitem__", [](const Shape& shape, long long index)
        {
            return shape(binding_utils::normalize_index(
                index,
                shape.rank(),
                "Shape index is out of bounds."));
        })
        .def("__iter__", [](const Shape& shape)
        {
            return py::make_iterator(shape.begin(), shape.end());
        }, py::keep_alive<0, 1>())
        .def("__eq__", [](const Shape& lhs, const Shape& rhs)
        {
            return lhs == rhs;
        })
        .def("__ne__", [](const Shape& lhs, const Shape& rhs)
        {
            return lhs != rhs;
        })
        .def("__repr__", [](const Shape& shape)
        {
            std::ostringstream os;
            os << shape;
            return os.str();
        });
}
