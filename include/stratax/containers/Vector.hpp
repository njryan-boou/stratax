#pragma once

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <utility>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/validation/Validation.hpp>

namespace stratax::container {

/** @brief Stores a rank-1 Stratax array in contiguous memory. */
template<typename T = double>
requires Numeric<T>
class Vector : public core::ArrayBase<T>
{
protected:
	using core::ArrayBase<T>::buffer_;
	using core::ArrayBase<T>::shape_;
	using core::ArrayBase<T>::strides_;

public:
	/** @brief Element type stored by the vector. */
	using value_type = T;

	/** @brief Rebinds the vector container to another element type. */
	template<typename U>
	using rebind = Vector<U>;

	/** @brief Creates a default vector with no logical dimensions. */
	Vector() noexcept = default;

	/** @brief Creates a vector with the given number of elements. */
	explicit Vector(std::size_t size)
	{
		shape_ = core::Shape({size}, core::Shape::allow_zero);
		strides_ = core::Strides(shape_);
		buffer_ = core::Buffer<T>(size);
	}

	/** @brief Creates a vector from a validated rank-1 shape. */
	explicit Vector(const core::Shape& shape)
	{
		shape_ = core::validation::require_rank(shape, 1, "Shape must be rank 1");
		strides_ = core::Strides(shape_);
		buffer_ = core::Buffer<T>(shape_.elements());
	}

	/** @brief Creates a vector and fills it with a value. */
	Vector(std::size_t size, const T& value)
	{
		shape_ = core::Shape({size}, core::Shape::allow_zero);
		strides_ = core::Strides(shape_);
		buffer_ = core::Buffer<T>(size, value);
	}

	/** @brief Creates a vector from an initializer list. */
	Vector(std::initializer_list<T> list)
	{
		shape_ = core::Shape({list.size()}, core::Shape::allow_zero);
		strides_ = core::Strides(shape_);
		buffer_ = core::Buffer<T>(list);
	}

	Vector(const Vector&) = default;
	Vector(Vector&&) noexcept = default;
	Vector& operator=(const Vector&) = default;
	Vector& operator=(Vector&&) noexcept = default;
	~Vector() = default;

	/** @brief Returns a flat element without bounds checking. */
	T& operator()(std::size_t index) noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns a flat element without bounds checking. */
	const T& operator()(std::size_t index) const noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns a flat element with bounds checking. */
	T& at(std::ptrdiff_t index)
	{
		const std::size_t normalized =
			core::validation::normalize_index(index, this->size(), "Vector index out of bounds.");
		return buffer_[normalized];
	}

	/** @brief Returns a flat element with bounds checking. */
	const T& at(std::ptrdiff_t index) const
	{
		const std::size_t normalized =
			core::validation::normalize_index(index, this->size(), "Vector index out of bounds.");
		return buffer_[normalized];
	}

};

} // namespace stratax::container
