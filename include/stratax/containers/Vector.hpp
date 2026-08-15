#pragma once

#include <cstddef>
#include <initializer_list>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>

namespace stratax::container {

template<typename T>
requires Numeric<T>
class Vector : public core::ArrayBase<T>
{
public:
	explicit Vector(std::size_t size)
		: core::ArrayBase<T>(core::Shape({size}))
	{}

	explicit Vector(const core::Shape& shape)
		: core::ArrayBase<T>(core::validation::require_rank(shape, 1, "Shape must be rank 1"))
	{}

	Vector(std::size_t size, const T& value)
		: core::ArrayBase<T>(core::Shape({size}), value)
	{}

	Vector(std::initializer_list<T> list)
		: core::ArrayBase<T>(core::Shape({list.size()}), core::Buffer<T>(list))
	{}

	Vector() : Vector(0) {}

	void swap(Vector& other) noexcept
	{
		using std::swap;

		swap(this->shape_, other.shape_);
		swap(this->strides_, other.strides_);
		swap(this->buffer_, other.buffer_);
	}

	friend void swap(Vector& lhs, Vector& rhs) noexcept
	{
		lhs.swap(rhs);
	}
};

} // namespace stratax::container
