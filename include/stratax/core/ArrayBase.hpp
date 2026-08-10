#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>

namespace stratax::core {

template<typename T>
class ArrayBase
{
protected:
	static_assert(Numeric<T>, "ArrayBase requires a Numeric value type.");

    Buffer<T> buffer_;
    Shape shape_;
    Strides strides_;
};

} // namespace stratax::core
