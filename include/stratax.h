#pragma once

#include <stratax/core/Buffer.hpp>
#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Config.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/Types.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/core/Slice.hpp>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

#include <stratax/algorithms/Creation.hpp>
#include <stratax/algorithms/Conversion.hpp>
#include <stratax/algorithms/Reshape.hpp>
#include <stratax/algorithms/Reductions.hpp>

#include <stratax/io/Print.hpp>

#include <stratax/ops/Comparison.hpp>
#include <stratax/ops/Arithmetic.hpp>
#include <stratax/ops/Bitwise.hpp>
#include <stratax/ops/Broadcasting.hpp>
#include <stratax/indexing/Indexing.hpp>
#include <stratax/indexing/Slicing.hpp>

namespace stratax {

using Shape = core::Shape;
using Slice = core::Slice;
using Strides = core::Strides;

template<typename T>
using Vector = container::Vector<T>;

template<typename T>
using Matrix = container::Matrix<T>;

template<typename T>
using Tensor = container::Tensor<T>;

namespace arrays {

using Shape = core::Shape;
using Slice = core::Slice;
using Strides = core::Strides;

}

namespace errors = ::Exceptions;

using ShapeError = errors::ShapeError;
using AxisError = errors::AxisError;
using BroadcastError = errors::BroadcastError;
using DimensionError = errors::DimensionError;
using IndexError = errors::IndexError;
using StrataxError = errors::StrataxError;
using TypeError = errors::TypeError;
using ZeroDivisionError = errors::ZeroDivisionError;

using ::flatten;
using ::reshape;

template<typename T>
using value_type_t = T;

template<typename T>
requires Numeric<T>
container::Tensor<T> zeros(const core::Shape& shape)
{
	return ::creation::zeros<T>(shape);
}

template<typename T>
requires Numeric<T>
container::Tensor<T> ones(const core::Shape& shape)
{
	return ::creation::ones<T>(shape);
}

template<typename T>
requires Numeric<T>
container::Tensor<T> full(const core::Shape& shape, const T& value)
{
	return ::creation::full<T>(shape, value);
}

template<typename T>
requires Numeric<T>
container::Tensor<T> identity(std::size_t size)
{
	return ::creation::identity<T>(size);
}

template<Array A>
auto sum(const A& arr)
{
	return ::reduction::sum(arr);
}

namespace creation = ::creation;
namespace reductions = ::reduction;

namespace transforms {

using ::flatten;
using ::reshape;

}

namespace conversions {

using ::to_matrix;

}

namespace slicing {

using ::slice;

}

}
