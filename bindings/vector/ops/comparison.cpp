#include <pybind11/pybind11.h>

#include <stratax/core/containers/Vector.hpp>
#include <stratax/core/ops/Comparison.hpp>

namespace py = pybind11;

namespace
{

using Vector = stratax::container::Vector<double>;

} // anonymous namespace

void bind_vector_comparison(py::class_<Vector>& cls)
{
    cls
        .def("__eq__", [](const Vector& lhs, const Vector& rhs) {
            return lhs == rhs;
        })
        .def("__ne__", [](const Vector& lhs, const Vector& rhs) {
            return lhs != rhs;
        });
}
