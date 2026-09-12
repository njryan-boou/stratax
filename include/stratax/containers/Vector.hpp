/** @file
 * @brief Rank-one owning arrays.
 */
#pragma once

#include <cstddef>
#include <initializer_list>

#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::container {

/**
 * @brief One-dimensional owning array of supported dtype values.
 *
 * Vector specializes core::ArrayBase for rank-one data. Elements are stored
 * contiguously, while the inherited shape and strides describe a row-major
 * layout whose sole stride is one. The class inherits flat checked and
 * unchecked access, iteration, metadata access, and modifiers from ArrayBase.
 *
 * @tparam T Element type satisfying the DType concept.
 *
 * @par Invariants
 * - `rank() == 1`, including for an empty vector.
 * - `shape()[0] == size()`.
 * - `strides()[0] == 1`.
 *
 * @par Example
 * @code
 * stratax::container::Vector<double> values{1.0, 2.0, 3.0};
 *
 * values[0] = 4.0; // unchecked access
 * values.at(-1);   // 3.0; negative indices count from the end
 * values.fill(2.0);
 * @endcode
 *
 * @see core::ArrayBase
 * @see core::Shape
 */
template<typename T>
requires DType<T>
class Vector : public core::ArrayBase<T>
{
public:
	/** @brief Stored element type inherited from ArrayBase. */
	using value_type = typename core::ArrayBase<T>::value_type;
	/** @brief Unsigned type used for element counts and indices. */
	using size_type = typename core::ArrayBase<T>::size_type;
	/** @brief Signed type used for checked indices and iterator distances. */
	using difference_type = typename core::ArrayBase<T>::difference_type;
	/** @brief Mutable element reference type. */
	using reference = typename core::ArrayBase<T>::reference;
	/** @brief Read-only element reference type. */
	using const_reference = typename core::ArrayBase<T>::const_reference;
	/** @brief Mutable element pointer type. */
	using pointer = typename core::ArrayBase<T>::pointer;
	/** @brief Read-only element pointer type. */
	using const_pointer = typename core::ArrayBase<T>::const_pointer;
	/** @brief Mutable contiguous random-access iterator type. */
	using iterator = typename core::ArrayBase<T>::iterator;
	/** @brief Read-only contiguous random-access iterator type. */
	using const_iterator = typename core::ArrayBase<T>::const_iterator;
	/** @brief Mutable reverse iterator type. */
	using reverse_iterator = typename core::ArrayBase<T>::reverse_iterator;
	/** @brief Read-only reverse iterator type. */
	using const_reverse_iterator = typename core::ArrayBase<T>::const_reverse_iterator;

	/**
	 * @brief Constructs a value-initialized vector with @p size elements.
	 * @param size Number of elements to allocate.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown while value-initializing a value_type.
	 * @complexity O(size).
	 */
	explicit Vector(size_type size)
		: core::ArrayBase<T>(core::Shape{size})
	{}

	/**
	 * @brief Constructs a value-initialized vector from a rank-one shape.
	 * @param shape Shape whose only dimension determines the vector size.
	 * @throws Exceptions::RankError If `shape.rank() != 1`.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown while value-initializing a value_type.
	 * @complexity O(shape.elements() + shape.rank()).
	 */
	explicit Vector(const core::Shape& shape)
		: core::ArrayBase<T>(shape)
	{
		if (shape.rank() != 1)
		{
			throw Exceptions::RankError("Vector requires a rank-1 shape.");
		}
	}

	/**
	 * @brief Constructs a vector containing @p size copies of @p value.
	 * @param size Number of elements to allocate.
	 * @param value Value copied into each element.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown by value_type's copy constructor.
	 * @complexity O(size).
	 */
	Vector(size_type size, const_reference value)
		: core::ArrayBase<T>(core::Shape{size}, value)
	{}

	/**
	 * @brief Constructs a vector by copying an initializer list.
	 * @param list Elements in vector order.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown by value_type's copy constructor.
	 * @complexity O(list.size()).
	 */
	Vector(std::initializer_list<value_type> list)
		: core::ArrayBase<T>(core::Shape{list.size()}, core::Buffer<value_type>(list))
	{}

	/**
	 * @brief Constructs an empty rank-one vector.
	 *
	 * The resulting shape is `{0}` rather than a rank-zero shape.
	 *
	 * @complexity O(1).
	 */
	Vector() : Vector(0) {}

	/** @brief Copies values and metadata into independent storage. */
	Vector(const Vector&) = default;
	/** @brief Replaces values and metadata with a deep copy; allocation failure leaves this object unchanged. */
	Vector& operator=(const Vector&) = default;

	/**
	 * @brief Transfers storage and leaves the source with shape `{0}`.
	 * @param other Source vector whose element allocation is transferred.
	 * @throws std::bad_alloc If allocating empty source metadata fails; the source is unchanged.
	 * @note Element values are not copied. This move is not noexcept.
	 * @complexity O(1).
	 */
	Vector(Vector&& other) : Vector() { swap(other); }

	/**
	 * @brief Transfers storage and preserves both vectors' rank invariants.
	 * @param other Source vector; a distinct source is left with shape `{0}`.
	 * @return Reference to this vector; self-move leaves it unchanged.
	 * @throws std::bad_alloc If empty metadata allocation fails; both objects are unchanged.
	 * @complexity O(size()) to destroy the previous elements, with O(1) ownership transfer.
	 */
	Vector& operator=(Vector&& other)
	{
		if (this != &other)
		{
			Vector moved(std::move(other));
			swap(moved);
		}
		return *this;
	}

	/**
	 * @brief Exchanges storage and layout metadata with @p other.
	 * @param other Vector to exchange with this vector.
	 * @complexity O(1).
	 */
	void swap(Vector& other) noexcept
	{
		core::ArrayBase<T>::swap(other);
	}

	/**
	 * @brief Exchanges two vectors using argument-dependent lookup.
	 * @param lhs First vector.
	 * @param rhs Second vector.
	 * @complexity O(1).
	 */
	friend void swap(Vector& lhs, Vector& rhs) noexcept
	{
		lhs.swap(rhs);
	}

};

} // namespace stratax::container
