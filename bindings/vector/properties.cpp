#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stratax/core/containers/Vector.hpp>
#include <stratax/core/io/Print.hpp>

#include <cstddef>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace
{

using Vector = stratax::container::Vector<double>;

} // anonymous namespace

void bind_vector_properties(py::class_<Vector>& cls)
{
    cls
        .def("size", &Vector::size)
        .def("rank", &Vector::rank)
        .def("empty", &Vector::empty)
        .def("shape", &Vector::shape, py::return_value_policy::reference_internal)
        .def("strides", [](const Vector& vector) {
            std::vector<std::size_t> values;
            const auto& strides = vector.strides();
            values.reserve(strides.rank());
            for (std::size_t stride : strides) {
                values.push_back(stride);
            }
            return values;
        })
        .def("fill", &Vector::fill, py::arg("value"))
        .def("tolist", [](const Vector& vector) {
            std::vector<double> values;
            values.reserve(vector.size());
            for (std::size_t i = 0; i < vector.size(); ++i) {
                values.push_back(vector[i]);
            }
            return values;
        })
        .def("__repr__", [](const Vector& vector) {
            std::ostringstream os;
            os << vector;
            return os.str();
        });
}
