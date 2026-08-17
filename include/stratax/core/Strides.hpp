#pragma once

#include <cstddef>
#include <limits>
#include <ostream>

#include "Buffer.hpp"
#include <stratax/core/Shape.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Normalize.hpp>

namespace stratax::core {

/**
 * @brief Stores row-major strides for a multidimensional shape.
 *
 * A stride is the number of contiguous elements skipped when an index in a
 * particular dimension is incremented by one. For a non-empty shape, the
 * final dimension has stride 1; each preceding stride is the product of the
 * dimensions to its right.
 *
 * Strides owns its values in contiguous storage. The values may be inspected
 * or modified through references, pointers, and iterators, although instances
 * constructed from a Shape initially contain canonical row-major strides.
 *
 * Construction rejects any stride product that cannot be represented by
 * `std::size_t`, preventing unsigned wraparound.
 *
 * @par Example
 * @code
 * const stratax::core::Shape shape{2, 3, 4};
 * const stratax::core::Strides strides(shape);
 *
 * // Row-major strides are (12, 4, 1).
 * const std::size_t offset =
 *     1 * strides[0] + 2 * strides[1] + 3 * strides[2]; // 23
 * @endcode
 *
 * @see Shape
 * @see Buffer
 */
class Strides
{
private:
	Buffer<std::size_t> buffer_;

public:
	/** @brief Type used to represent each stride. */
	using value_type = std::size_t;
	/** @brief Unsigned type used for ranks and stride indices. */
	using size_type = std::size_t;
	/** @brief Signed type used for checked indices and iterator distances. */
	using difference_type = std::ptrdiff_t;
	/** @brief Mutable reference to a stride. */
	using reference = value_type&;
	/** @brief Read-only reference to a stride. */
	using const_reference = const value_type&;
	/** @brief Mutable pointer to a stride. */
	using pointer = value_type*;
	/** @brief Read-only pointer to a stride. */
	using const_pointer = const value_type*;
	/** @brief Mutable contiguous iterator over strides. */
	using iterator = Buffer<value_type>::iterator;
	/** @brief Read-only contiguous iterator over strides. */
	using const_iterator = Buffer<value_type>::const_iterator;
	/** @brief Mutable iterator over strides in reverse order. */
	using reverse_iterator = Buffer<value_type>::reverse_iterator;
	/** @brief Read-only iterator over strides in reverse order. */
	using const_reverse_iterator = Buffer<value_type>::const_reverse_iterator;

	/**
	 * @brief Constructs an empty stride sequence.
	 *
	 * The resulting object has rank zero and owns no allocation.
	 */
	Strides() noexcept = default;

	/**
	 * @brief Computes canonical row-major strides for @p shape.
	 *
	 * The resulting rank equals `shape.rank()`. An empty shape produces empty
	 * strides. For a non-empty shape, `back()` is 1 and every earlier value is
	 * computed from the dimensions to its right.
	 *
	 * @param shape Shape whose row-major layout is represented.
	 * @throws std::bad_alloc If allocation of the stride storage fails.
	 * @throws Exceptions::DimensionError If a stride product exceeds
	 *         `std::size_t`.
	 */
	explicit Strides(const Shape& shape)
		: buffer_(shape.rank())
	{
		if (shape.empty()) {
			return;
		}

		buffer_[shape.rank() - 1] = 1;

		for (std::size_t i = shape.rank() - 1; i > 0; --i) {
			if (shape[i] != 0 &&
				buffer_[i] > std::numeric_limits<std::size_t>::max() / shape[i]) {
				throw Exceptions::DimensionError(
					"Stride value exceeds the maximum representable size.");
			}
			else {
				buffer_[i - 1] = buffer_[i] * shape[i];
			}
		}
	}

	/**
	 * @brief Returns a pointer to the contiguous stride storage.
	 *
	 * @return Pointer to the first stride, or `nullptr` when empty.
	 *
	 * The pointer remains valid until this object is assigned to, moved from,
	 * swapped, or destroyed.
	 */
	[[nodiscard]] pointer data() noexcept {return buffer_.data();}

	/**
	 * @brief Returns a const pointer to the contiguous stride storage.
	 *
	 * @return Const pointer to the first stride, or `nullptr` when empty.
	 *
	 * The pointer remains valid until this object is assigned to, moved from,
	 * swapped, or destroyed.
	 */
	[[nodiscard]] const_pointer data() const noexcept {return buffer_.data();}

	/**
	 * @brief Returns a mutable reference to the first stride.
	 *
	 * @return Reference to the stride for the first dimension.
	 * @throws Exceptions::IndexError If this stride sequence is empty.
	 */
	[[nodiscard]] reference front()
	{
		if (empty()) {
			throw Exceptions::IndexError("Strides front cannot be accessed when the strides are empty.");
		}
		return buffer_.front();
	}

	/**
	 * @brief Returns a const reference to the first stride.
	 *
	 * @return Const reference to the stride for the first dimension.
	 * @throws Exceptions::IndexError If this stride sequence is empty.
	 */
	[[nodiscard]] const_reference front() const
	{
		if (empty()) {
			throw Exceptions::IndexError("Strides front cannot be accessed when the strides are empty.");
		}
		return buffer_.front();
	}

	/**
	 * @brief Returns a mutable reference to the final stride.
	 *
	 * For canonical non-empty row-major strides, this value is 1.
	 *
	 * @return Reference to the stride for the final dimension.
	 * @throws Exceptions::IndexError If this stride sequence is empty.
	 */
	[[nodiscard]] reference back()
	{
		if (empty()) {
			throw Exceptions::IndexError("Strides back cannot be accessed when the strides are empty.");
		}
		return buffer_.back();
	}

	/**
	 * @brief Returns a const reference to the final stride.
	 *
	 * For canonical non-empty row-major strides, this value is 1.
	 *
	 * @return Const reference to the stride for the final dimension.
	 * @throws Exceptions::IndexError If this stride sequence is empty.
	 */
	[[nodiscard]] const_reference back() const
	{
		if (empty()) {
			throw Exceptions::IndexError("Strides back cannot be accessed when the strides are empty.");
		}
		return buffer_.back();
	}

	/** @brief Returns a mutable iterator to the first stride. */
	iterator begin() noexcept {return buffer_.begin();}
	/** @brief Returns a const iterator to the first stride. */
	const_iterator begin() const noexcept {return buffer_.begin();}
	/** @brief Returns a const iterator to the first stride. */
	const_iterator cbegin() const noexcept {return buffer_.cbegin();}
	/** @brief Returns a mutable iterator one past the final stride. */
	iterator end() noexcept {return buffer_.end();}
	/** @brief Returns a const iterator one past the final stride. */
	const_iterator end() const noexcept {return buffer_.end();}
	/** @brief Returns a const iterator one past the final stride. */
	const_iterator cend() const noexcept {return buffer_.cend();}
	/** @brief Returns a mutable reverse iterator to the final stride. */
	reverse_iterator rbegin() noexcept {return buffer_.rbegin();}
	/** @brief Returns a const reverse iterator to the final stride. */
	const_reverse_iterator rbegin() const noexcept {return buffer_.rbegin();}
	/** @brief Returns a const reverse iterator to the final stride. */
	const_reverse_iterator crbegin() const noexcept {return buffer_.crbegin();}
	/** @brief Returns the past-the-end iterator for mutable reverse traversal. */
	reverse_iterator rend() noexcept {return buffer_.rend();}
	/** @brief Returns the past-the-end iterator for const reverse traversal. */
	const_reverse_iterator rend() const noexcept {return buffer_.rend();}
	/** @brief Returns the past-the-end iterator for const reverse traversal. */
	const_reverse_iterator crend() const noexcept {return buffer_.crend();}

	/**
	 * @brief Reports whether no stride values are stored.
	 *
	 * @return `true` when rank() is zero; otherwise `false`.
	 */
	[[nodiscard]] bool empty() const noexcept {return buffer_.empty();}

	/**
	 * @brief Returns the number of represented dimensions.
	 *
	 * @return Number of stored stride values.
	 */
	[[nodiscard]] size_type rank() const noexcept {return buffer_.size();}

	/**
	 * @brief Returns a stride without bounds checking.
	 *
	 * @param index Zero-based dimension index.
	 * @return Const reference to the stride at @p index.
	 * @pre `index < rank()`; otherwise behavior is undefined.
	 */
	[[nodiscard]] const_reference operator[](size_type index) const noexcept {return buffer_[index];}

	/**
	 * @brief Returns a stride using checked, Python-style indexing.
	 *
	 * Non-negative indices count from the beginning. Negative indices count
	 * backward from the end, so `at(-1)` returns the final stride.
	 *
	 * @param index Dimension index in the range `[-rank(), rank())`.
	 * @return Const reference to the normalized stride position.
	 * @throws Exceptions::IndexError If @p index is outside the valid range.
	 */
	[[nodiscard]] const_reference at(difference_type index) const {return buffer_[indexing::normalize_index(index, rank())];}

	/**
	 * @brief Exchanges the stored stride sequences with @p other.
	 *
	 * This operation is constant time and does not copy individual values.
	 * Existing pointers and iterators remain associated with their original
	 * values, which become owned by the other Strides object.
	 *
	 * @param other Object whose stride storage is exchanged with this object.
	 */
	void swap(Strides& other) noexcept {buffer_.swap(other.buffer_);}

	/**
	 * @brief Compares two stride sequences element by element.
	 *
	 * @param other Strides object to compare with.
	 * @return `true` if both objects have the same rank and values.
	 */
	[[nodiscard]] bool operator==(const Strides& other) const noexcept
	{
		if (rank() != other.rank()) {
			return false;
		}

		for (size_type i = 0; i < rank(); ++i) {
			if (buffer_[i] != other.buffer_[i]) {
				return false;
			}
		}

		return true;
	}

};

/**
 * @brief Writes stride values in tuple notation.
 *
 * Values are comma-separated and enclosed in parentheses. A one-dimensional
 * sequence includes a trailing comma, for example `(1,)`; an empty sequence
 * is written as `()`.
 *
 * @param os Output stream receiving the formatted representation.
 * @param stride Stride sequence to write.
 * @return Reference to @p os.
 */
inline std::ostream& operator<<(std::ostream& os, const Strides& stride)
{
	os << "(";

	bool first = true;
	for (Strides::size_type i = 0; i < stride.rank(); ++i)
{
    if (i != 0)
    {
        os << ", ";
    }

    os << stride[i];
}

	if (stride.rank() == 1)
	{
		os << ",";
	}

	os << ")";

	return os;
}

}
