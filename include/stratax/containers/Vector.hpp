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
template<typename T>
requires Numeric<T>
class Vector : public core::ArrayBase<T>
{
protected:
	using core::ArrayBase<T>::allocate_with_size;
	using core::ArrayBase<T>::buffer_;
	using core::ArrayBase<T>::checked_flat_ref;
	using core::ArrayBase<T>::allocate_from_shape;
	using core::ArrayBase<T>::set_shape_and_strides;
	using core::ArrayBase<T>::shape_;
	using core::ArrayBase<T>::strides_;
	using core::ArrayBase<T>::normalize_flat_index;

public:
	/** @brief Rebinds the vector container to another element type. */
	template<typename U>
	using rebind = Vector<U>;

	/** @brief Creates a default vector with no logical dimensions. */
	Vector() noexcept = default;

	/** @brief Creates a vector with the given number of elements. */
	explicit Vector(std::size_t size)
	{
		set_shape_and_strides(core::Shape({size}, core::Shape::allow_zero));
		allocate_with_size(size);
	}

	/** @brief Creates a vector from a validated rank-1 shape. */
	explicit Vector(const core::Shape& shape)
	{
		set_shape_and_strides(core::validation::require_rank(shape, 1, "Shape must be rank 1"));
		allocate_from_shape();
	}

	/** @brief Creates a vector and fills it with a value. */
	Vector(std::size_t size, const T& value)
	{
		set_shape_and_strides(core::Shape({size}, core::Shape::allow_zero));
		allocate_with_size(size, value);
	}

	/** @brief Creates a vector from an initializer list. */
	Vector(std::initializer_list<T> list)
	{
		set_shape_and_strides(core::Shape({list.size()}, core::Shape::allow_zero));
		buffer_ = core::Buffer<T>(list);
	}

	Vector(const Vector&) = default;
	Vector(Vector&&) noexcept = default;
	Vector& operator=(const Vector&) = default;
	Vector& operator=(Vector&&) noexcept = default;
	~Vector() = default;

	/** @brief Returns a flat element without bounds checking. */
	T& operator()(std::size_t index)
	{
		return buffer_[index];
	}

	/** @brief Returns a flat element without bounds checking. */
	const T& operator()(std::size_t index) const
	{
		return buffer_[index];
	}

	/** @brief Returns a flat element with bounds checking. */
	T& at(std::ptrdiff_t index)
	{
		return checked_flat_ref(index);
	}

	/** @brief Returns a flat element with bounds checking. */
	const T& at(std::ptrdiff_t index) const
	{
		return checked_flat_ref(index);
	}

};

} // namespace stratax::container
