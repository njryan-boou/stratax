#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <vector>
#include <limits>

#include <stratax/core/Buffer.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Normalize.hpp>

namespace stratax::core {

/**
 * @brief Stores the dimensions of a multidimensional array.
 *
 * Dimensions are stored in contiguous, read-only order. The number of stored
 * dimensions is the shape's rank, and elements() returns their product when
 * that product can be represented by `std::size_t`.
 *
 * @par Example
 * @code
 * const stratax::core::Shape shape{2, 3, 4};
 *
 * shape.rank();     // 3
 * shape.elements(); // 24
 * shape.at(-1);     // 4
 * @endcode
 */
class Shape
{
private:
	Buffer<std::size_t> dims_;

public:
	/** @brief Read-only contiguous iterator over dimensions. */
	using const_iterator = Buffer<std::size_t>::const_iterator;
	/** @brief Read-only iterator over dimensions in reverse order. */
	using const_reverse_iterator = Buffer<std::size_t>::const_reverse_iterator;

	/**
	 * @brief Constructs an empty, rank-zero shape.
	 * @complexity O(1).
	 */
	Shape() noexcept = default;

	/**
	 * @brief Constructs a shape from dimensions in outermost-to-innermost order.
	 *
	 * @param dims Dimensions to copy into the shape.
	 * @throws std::bad_alloc If dimension storage cannot be allocated.
	 * @complexity O(dims.size()).
	 */
	Shape(std::initializer_list<std::size_t> dims)
		: dims_(dims)
	{}

	/**
	 * @brief Constructs a shape by copying a vector of dimensions.
	 *
	 * @param dims Dimensions in outermost-to-innermost order.
	 * @throws std::bad_alloc If dimension storage cannot be allocated.
	 * @complexity O(dims.size()).
	 */
	Shape(const std::vector<std::size_t>& dims)
		: dims_(dims.size())
	{
		std::copy(dims.begin(), dims.end(), dims_.begin());
	}

	/**
	 * @brief Computes the total number of elements described by the shape.
	 *
	 * Returns zero for a rank-zero shape or for any shape containing a zero
	 * dimension. Multiplication is checked to prevent unsigned overflow.
	 *
	 * @return Product of all dimensions, or zero for an empty/zero-sized shape.
	 * @throws Exceptions::DimensionError If the product exceeds `std::size_t`.
	 * @complexity O(rank()).
	 */
	[[nodiscard]] std::size_t elements() const
	{
		if (empty())
		{
			return 0;
		}
		std::size_t prod = 1;
		for (std::size_t dim : dims_)
		{
			if (dim == 0)
			{
				return 0;
			}

			if (prod > std::numeric_limits<std::size_t>::max() / dim) {
				throw Exceptions::DimensionError(
				"Shape element count exceeds the maximum representable size.");
			}

			prod *= dim;
		}

		return prod;
	}

	/**
	 * @brief Returns the number of dimensions.
	 * @return Number of stored dimension values.
	 * @complexity O(1).
	 */
	[[nodiscard]] std::size_t rank() const noexcept
	{
		return dims_.size();
	}

	/**
	 * @brief Returns a dimension without bounds checking.
	 * @param index Zero-based dimension index.
	 * @return Const reference to the dimension at @p index.
	 * @pre `index < rank()`; otherwise behavior is undefined.
	 * @complexity O(1).
	 */
	[[nodiscard]] const std::size_t& operator[](std::size_t index) const noexcept
	{
		return dims_[index];
	}

	/**
	 * @brief Returns a dimension using checked, Python-style indexing.
	 *
	 * Negative indices count backward from the final dimension, so `at(-1)`
	 * returns the last dimension.
	 *
	 * @param index Dimension index in the range `[-rank(), rank())`.
	 * @return Const reference to the normalized dimension.
	 * @throws Exceptions::IndexError If @p index is outside the valid range.
	 * @complexity O(1).
	 */
	[[nodiscard]] const std::size_t& at(std::ptrdiff_t index) const
	{
		return dims_[stratax::indexing::normalize_index(index, rank())];
	}

	/**
	 * @brief Reports whether the shape has rank zero.
	 * @return `true` if no dimensions are stored; otherwise `false`.
	 * @complexity O(1).
	 */
	[[nodiscard]] bool empty() const noexcept
	{
		return dims_.empty();
	}

	/**
	 * @brief Compares two shapes dimension by dimension.
	 * @param other Shape to compare with.
	 * @return `true` if both shapes have identical rank and dimensions.
	 * @complexity O(rank()).
	 */
	[[nodiscard]] bool operator==(const Shape& other) const noexcept
	{
		if (rank() != other.rank())
		{
			return false;
		}
		for (std::size_t i = 0; i < rank(); ++i)
		{
			if (dims_[i] != other.dims_[i])
			{
				return false;
			}
		}
		return true;
	}

	/** @brief Returns a const iterator to the first dimension. @complexity O(1). */
	const_iterator begin() const noexcept {return dims_.begin();}
	/** @brief Returns a const iterator past the final dimension. @complexity O(1). */
	const_iterator end() const noexcept {return dims_.end();}
	/** @brief Returns a const iterator to the first dimension. @complexity O(1). */
	const_iterator cbegin() const noexcept {return dims_.cbegin();}
	/** @brief Returns a const iterator past the final dimension. @complexity O(1). */
	const_iterator cend() const noexcept {return dims_.cend();}
	/** @brief Returns a const reverse iterator to the final dimension. @complexity O(1). */
	const_reverse_iterator rbegin() const noexcept {return dims_.rbegin();}
	/** @brief Returns a const reverse iterator to the final dimension. @complexity O(1). */
	const_reverse_iterator crbegin() const noexcept {return dims_.crbegin();}
	/** @brief Returns the past-the-end reverse iterator. @complexity O(1). */
	const_reverse_iterator rend() const noexcept {return dims_.rend();}
	/** @brief Returns the past-the-end const reverse iterator. @complexity O(1). */
	const_reverse_iterator crend() const noexcept {return dims_.crend();}

	/**
	 * @brief Exchanges dimension storage with another shape.
	 * @param other Shape whose dimensions are exchanged with this shape.
	 * @complexity O(1).
	 */
	void swap(Shape& other) noexcept {dims_.swap(other.dims_);}

};

/**
 * @brief Writes a shape using tuple notation.
 *
 * A one-dimensional shape includes a trailing comma, such as `(5,)`, and an
 * empty shape is written as `()`.
 *
 * @param os Output stream receiving the representation.
 * @param shape Shape to write.
 * @return Reference to @p os.
 * @complexity O(shape.rank()).
 */
inline std::ostream& operator<<(std::ostream& os, const Shape& shape)
{
	os << "(";

	bool first = true;
	for (std::size_t dim : shape)
	{
		if (!first)
			os << ", ";

		os << dim;
		first = false;
	}

	if (shape.rank() == 1)
	{
		os << ",";
	}

	os << ")";

	return os;
}

}
