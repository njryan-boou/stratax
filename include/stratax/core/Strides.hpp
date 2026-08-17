#pragma once

#include <cstddef>
#include <limits>
#include <ostream>

#include "Buffer.hpp"
#include <stratax/core/Shape.hpp>
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
 * If a product needed to represent a stride exceeds `std::size_t`, that
 * stride is stored as `std::numeric_limits<std::size_t>::max()` instead of
 * allowing unsigned wraparound.
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
	using iterator = Buffer<std::size_t>::iterator;
	using const_iterator = Buffer<std::size_t>::const_iterator;
	using reverse_iterator = Buffer<std::size_t>::reverse_iterator;
	using const_reverse_iterator = Buffer<std::size_t>::const_reverse_iterator;

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
	 * computed from the dimensions to its right. A product that cannot be
	 * represented by `std::size_t` is replaced by its maximum value.
	 *
	 * @param shape Shape whose row-major layout is represented.
	 * @throws std::bad_alloc If allocation of the stride storage fails.
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
				buffer_[i - 1] = std::numeric_limits<std::size_t>::max();
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
	[[nodiscard]] std::size_t* data() noexcept
	{
		return buffer_.data();
	}

	/**
	 * @brief Returns a const pointer to the contiguous stride storage.
	 *
	 * @return Const pointer to the first stride, or `nullptr` when empty.
	 *
	 * The pointer remains valid until this object is assigned to, moved from,
	 * swapped, or destroyed.
	 */
	[[nodiscard]] const std::size_t* data() const noexcept
	{
		return buffer_.data();
	}

	/**
	 * @brief Returns a mutable reference to the first stride.
	 *
	 * @return Reference to the stride for the first dimension.
	 * @throws Exceptions::IndexError If this stride sequence is empty.
	 */
	[[nodiscard]] std::size_t& front()
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
	[[nodiscard]] const std::size_t& front() const
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
	[[nodiscard]] std::size_t& back()
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
	[[nodiscard]] const std::size_t& back() const
	{
		if (empty()) {
			throw Exceptions::IndexError("Strides back cannot be accessed when the strides are empty.");
		}
		return buffer_.back();
	}

	/** @brief Returns a mutable iterator to the first stride. */
	[[nodiscard]] iterator begin() noexcept
	{
		return buffer_.begin();
	}

	/** @brief Returns a const iterator to the first stride. */
	[[nodiscard]] const_iterator begin() const noexcept
	{
		return buffer_.begin();
	}

	/** @brief Returns a const iterator to the first stride. */
	[[nodiscard]] const_iterator cbegin() const noexcept
	{
		return buffer_.cbegin();
	}

	/** @brief Returns a mutable iterator one past the final stride. */
	[[nodiscard]] iterator end() noexcept
	{
		return buffer_.end();
	}

	/** @brief Returns a const iterator one past the final stride. */
	[[nodiscard]] const_iterator end() const noexcept
	{
		return buffer_.end();
	}

	/** @brief Returns a const iterator one past the final stride. */
	[[nodiscard]] const_iterator cend() const noexcept
	{
		return buffer_.cend();
	}

	/** @brief Returns a mutable reverse iterator to the final stride. */
	[[nodiscard]] reverse_iterator rbegin() noexcept
	{
		return buffer_.rbegin();
	}

	/** @brief Returns a const reverse iterator to the final stride. */
	[[nodiscard]] const_reverse_iterator rbegin() const noexcept
	{
		return buffer_.rbegin();
	}

	/** @brief Returns a const reverse iterator to the final stride. */
	[[nodiscard]] const_reverse_iterator crbegin() const noexcept
	{
		return buffer_.crbegin();
	}

	/** @brief Returns the past-the-end iterator for mutable reverse traversal. */
	[[nodiscard]] reverse_iterator rend() noexcept
	{
		return buffer_.rend();
	}

	/** @brief Returns the past-the-end iterator for const reverse traversal. */
	[[nodiscard]] const_reverse_iterator rend() const noexcept
	{
		return buffer_.rend();
	}

	/** @brief Returns the past-the-end iterator for const reverse traversal. */
	[[nodiscard]] const_reverse_iterator crend() const noexcept
	{
		return buffer_.crend();
	}

	/**
	 * @brief Reports whether no stride values are stored.
	 *
	 * @return `true` when rank() is zero; otherwise `false`.
	 */
	[[nodiscard]] bool empty() const noexcept
	{
		return buffer_.empty();
	}

	/**
	 * @brief Returns the number of represented dimensions.
	 *
	 * @return Number of stored stride values.
	 */
	[[nodiscard]] std::size_t rank() const noexcept
	{
		return buffer_.size();
	}

	/**
	 * @brief Returns a stride without bounds checking.
	 *
	 * @param index Zero-based dimension index.
	 * @return Const reference to the stride at @p index.
	 * @pre `index < rank()`; otherwise behavior is undefined.
	 */
	const std::size_t& operator[](std::size_t index) const noexcept
	{
		return buffer_[index];
	}

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
	const std::size_t& at(std::ptrdiff_t index) const
{
    return buffer_[stratax::indexing::normalize_index(index, rank())];
}

	/**
	 * @brief Exchanges the stored stride sequences with @p other.
	 *
	 * This operation is constant time and does not copy individual values.
	 * Existing pointers and iterators remain associated with their original
	 * values, which become owned by the other Strides object.
	 *
	 * @param other Object whose stride storage is exchanged with this object.
	 */
	void swap(Strides& other) noexcept
	{
		buffer_.swap(other.buffer_);
	}

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

		for (std::size_t i = 0; i < rank(); ++i) {
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
	for (std::size_t i = 0; i < stride.rank(); ++i)
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
