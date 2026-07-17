#include <pybind11/pybind11.h>

#include <stratax/core/containers/Shape.hpp>

namespace py = pybind11;

void bind_shape_constructors(py::class_<stratax::core::Shape>&);
void bind_shape_properties(py::class_<stratax::core::Shape>&);

void bind_shape(py::module_& m)
{
    using Shape = stratax::core::Shape;

    py::class_<Shape> cls(m, "_Shape");

    bind_shape_constructors(cls);
    bind_shape_properties(cls);
}
