#pragma once

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <utility>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/validation/Validation.hpp>

namespace stratax::container {

/** @brief Stores a rank-1 Stratax array in contiguous memory. */
template<typename T>
requires Numeric<T>
class Vector
{
private:
	core::Shape shape_;
	core::Strides strides_;
	core::Buffer<T> buffer_;

public:
	/** @brief Element type stored by the vector. */
	using value_type = T;

	/** @brief Mutable iterator over vector elements. */
	using iterator = typename core::Buffer<T>::iterator;

	/** @brief Const iterator over vector elements. */
	using const_iterator = typename core::Buffer<T>::const_iterator;

	/** @brief Mutable reverse iterator over vector elements. */
	using reverse_iterator = typename core::Buffer<T>::reverse_iterator;

	/** @brief Const reverse iterator over vector elements. */
	using const_reverse_iterator = typename core::Buffer<T>::const_reverse_iterator;

	/** @brief Rebinds the vector container to another element type. */
	template<typename U>
	using rebind = Vector<U>;

	/** @brief Creates a default vector with no logical dimensions. */
	Vector() noexcept = default;

	/** @brief Creates a vector with the given number of elements. */
	explicit Vector(std::size_t size)
		: shape_({size}, core::Shape::allow_zero),
		  strides_(shape_),
		  buffer_(size)
	{
	}

	/** @brief Creates a vector from a validated rank-1 shape. */
	explicit Vector(const core::Shape& shape)
		: shape_(core::validation::require_rank(shape, 1, "Shape must be rank 1")),
		  strides_(shape_),
		  buffer_(shape_.elements())
	{
	}

	/** @brief Creates a vector and fills it with a value. */
	Vector(std::size_t size, const T& value)
		: shape_({size}, core::Shape::allow_zero),
		  strides_(shape_),
		  buffer_(size, value)
	{
	}

	/** @brief Creates a vector from an initializer list. */
	Vector(std::initializer_list<T> list)
		: shape_({list.size()}, core::Shape::allow_zero),
		  strides_(shape_),
		  buffer_(list)
	{
	}

	/** @brief Creates a copy of another vector. */
	Vector(const Vector&) = default;

	/** @brief Transfers ownership from another vector. */
	Vector(Vector&&) noexcept = default;

	/** @brief Replaces this vector with a copy of another vector. */
	Vector& operator=(const Vector&) = default;

	/** @brief Replaces this vector by taking ownership from another vector. */
	Vector& operator=(Vector&&) noexcept = default;

	/** @brief Destroys the vector. */
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

	/** @brief Returns a flat element without bounds checking. */
	T& operator[](std::size_t index) noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns a flat element without bounds checking. */
	const T& operator[](std::size_t index) const noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns a flat element with bounds checking. */
	T& at(std::ptrdiff_t index)
	{
		const std::size_t normalized =
			core::validation::normalize_index(index, size(), "Vector index out of bounds.");
		return buffer_[normalized];
	}

	/** @brief Returns a flat element with bounds checking. */
	const T& at(std::ptrdiff_t index) const
	{
		const std::size_t normalized =
			core::validation::normalize_index(index, size(), "Vector index out of bounds.");
		return buffer_[normalized];
	}

	/** @brief Returns the first element. */
	T& front()
	{
		return buffer_.front();
	}

	/** @brief Returns the first element as a const reference. */
	const T& front() const
	{
		return buffer_.front();
	}

	/** @brief Returns the last element. */
	T& back()
	{
		return buffer_.back();
	}

	/** @brief Returns the last element as a const reference. */
	const T& back() const
	{
		return buffer_.back();
	}

	/** @brief Returns the number of stored elements. */
	[[nodiscard]] std::size_t size() const noexcept
	{
		return shape_.elements();
	}

	/** @brief Returns the vector rank. */
	[[nodiscard]] std::size_t rank() const noexcept
	{
		return shape_.rank();
	}

	/** @brief Returns whether the vector has no elements. */
	[[nodiscard]] bool empty() const noexcept
	{
		return buffer_.empty();
	}

	/** @brief Returns the vector shape. */
	const stratax::core::Shape& shape() const noexcept
	{
		return shape_;
	}

	/** @brief Returns the vector strides. */
	const stratax::core::Strides& strides() const noexcept
	{
		return strides_;
	}

	/** @brief Returns the raw data pointer. */
	[[nodiscard]] T* data() noexcept
	{
		return buffer_.data();
	}

	/** @brief Returns the raw data pointer as a const pointer. */
	[[nodiscard]] const T* data() const noexcept
	{
		return buffer_.data();
	}

	/** @brief Returns an iterator to the first element. */
	[[nodiscard]] iterator begin() noexcept
	{
		return buffer_.begin();
	}

	/** @brief Returns a const iterator to the first element. */
	[[nodiscard]] const_iterator begin() const noexcept
	{
		return buffer_.begin();
	}

	/** @brief Returns a const iterator to the first element. */
	[[nodiscard]] const_iterator cbegin() const noexcept
	{
		return buffer_.cbegin();
	}

	/** @brief Returns an iterator one past the last element. */
	[[nodiscard]] iterator end() noexcept
	{
		return buffer_.end();
	}

	/** @brief Returns a const iterator one past the last element. */
	[[nodiscard]] const_iterator end() const noexcept
	{
		return buffer_.end();
	}

	/** @brief Returns a const iterator one past the last element. */
	[[nodiscard]] const_iterator cend() const noexcept
	{
		return buffer_.cend();
	}

	/** @brief Returns a reverse iterator to the last element. */
	[[nodiscard]] reverse_iterator rbegin() noexcept
	{
		return buffer_.rbegin();
	}

	/** @brief Returns a const reverse iterator to the last element. */
	[[nodiscard]] const_reverse_iterator rbegin() const noexcept
	{
		return buffer_.rbegin();
	}

	/** @brief Returns a const reverse iterator to the last element. */
	[[nodiscard]] const_reverse_iterator crbegin() const noexcept
	{
		return buffer_.crbegin();
	}

	/** @brief Returns a reverse iterator before the first element. */
	[[nodiscard]] reverse_iterator rend() noexcept
	{
		return buffer_.rend();
	}

	/** @brief Returns a const reverse iterator before the first element. */
	[[nodiscard]] const_reverse_iterator rend() const noexcept
	{
		return buffer_.rend();
	}

	/** @brief Returns a const reverse iterator before the first element. */
	[[nodiscard]] const_reverse_iterator crend() const noexcept
	{
		return buffer_.crend();
	}

	/** @brief Fills every element with the same value. */
	void fill(const T& value)
	{
		buffer_.fill(value);
	}

	/** @brief Swaps the contents of two vectors. */
	void swap(Vector& other) noexcept
	{
		using std::swap;

		swap(shape_, other.shape_);
		swap(strides_, other.strides_);
		swap(buffer_, other.buffer_);
	}
};

} // namespace stratax::container
