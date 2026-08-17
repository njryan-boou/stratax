#pragma once

#include <cstddef>
#include <initializer_list>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::container {

template<typename T>
requires Numeric<T>
class Vector : public core::ArrayBase<T>
{
public:
	explicit Vector(std::size_t size)
		: core::ArrayBase<T>(core::Shape{size})
	{}

	explicit Vector(const core::Shape& shape)
		: core::ArrayBase<T>(validate_shape(shape))
	{}

	Vector(std::size_t size, const T& value)
		: core::ArrayBase<T>(core::Shape{size}, value)
	{}

	Vector(std::initializer_list<T> list)
		: core::ArrayBase<T>(core::Shape{list.size()}, core::Buffer<T>(list))
	{}

	Vector() : Vector(0) {}

	void swap(Vector& other) noexcept
	{
		core::ArrayBase<T>::swap(other);
	}

	friend void swap(Vector& lhs, Vector& rhs) noexcept
	{
		lhs.swap(rhs);
	}

private:
	static const core::Shape& validate_shape(const core::Shape& shape)
	{
		if (shape.rank() != 1) {
			throw Exceptions::ShapeError(
				"Vector shape must be rank 1.");
		}

		return shape;
	}
};

} // namespace stratax::container
