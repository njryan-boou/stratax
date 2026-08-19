#pragma once

#include <cstddef>
#include <utility>

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Indexing.hpp>
#include <stratax/indexing/Normalize.hpp>
#include <stratax/core/DTypeTraits.hpp>

namespace stratax::core {

/**
 * @brief Shared owning storage and layout base for Stratax array containers.
 *
 * ArrayBase keeps an element Buffer synchronized with Shape metadata and its
 * derived row-major Strides. It supplies the common one-dimensional container
 * interface used by Vector, Matrix, and Tensor; derived classes provide rank
 * constraints and multidimensional indexing APIs.
 *
 * The class is not directly constructible by users because its constructors
 * are protected.
 *
 * @tparam T Type of each stored array element.
 *
 * @par Invariants
 * - `size() == shape().elements()`.
 * - `rank() == shape().rank() == strides().rank()`.
 * - `strides()` represents the row-major layout of `shape()`.
 *
 * @see Buffer
 * @see Shape
 * @see Strides
 */
template<typename T>
class ArrayBase
{
public:
	/** @brief Stored element type. */
	using value_type = typename Buffer<T>::value_type;
	/** @brief Unsigned type used for element counts and indices. */
	using size_type = typename Buffer<value_type>::size_type;
	/** @brief Signed type used for checked indices and iterator distances. */
	using difference_type = typename Buffer<value_type>::difference_type;
	/** @brief Mutable element reference type. */
	using reference = typename Buffer<value_type>::reference;
	/** @brief Read-only element reference type. */
	using const_reference = typename Buffer<value_type>::const_reference;
	/** @brief Mutable element pointer type. */
	using pointer = typename Buffer<value_type>::pointer;
	/** @brief Read-only element pointer type. */
	using const_pointer = typename Buffer<value_type>::const_pointer;
	/** @brief Mutable contiguous random-access iterator type. */
	using iterator = typename Buffer<value_type>::iterator;
	/** @brief Read-only contiguous random-access iterator type. */
	using const_iterator = typename Buffer<value_type>::const_iterator;
	/** @brief Mutable reverse iterator type. */
	using reverse_iterator = typename Buffer<value_type>::reverse_iterator;
	/** @brief Read-only reverse iterator type. */
	using const_reverse_iterator = typename Buffer<value_type>::const_reverse_iterator;

	[[nodiscard]] static constexpr std::string_view dtype() noexcept
	{
		return stratax::core::DTypeTraits<value_type>::name;
	}

	/** @brief Returns the number of stored elements. @complexity O(1). */
	[[nodiscard]] size_type size() const noexcept {return buffer_.size();}

	/** @brief Reports whether no elements are stored. @complexity O(1). */
	[[nodiscard]] bool empty() const noexcept {return buffer_.empty();}

	/** @brief Returns the number of logical dimensions. @complexity O(1). */
	[[nodiscard]] size_type rank() const noexcept {return shape_.rank();}

	/** @brief Returns the logical shape metadata. @complexity O(1). */
	[[nodiscard]] const Shape& shape() const noexcept {return shape_;}

	/** @brief Returns the row-major stride metadata. @complexity O(1). */
	[[nodiscard]] const Strides& strides() const noexcept {return strides_;}

	/**
	 * @brief Returns a pointer to contiguous mutable element storage.
	 * @return First element address, or `nullptr` when empty.
	 * @complexity O(1).
	 */
	[[nodiscard]] pointer data() noexcept {return buffer_.data();}
	/**
	 * @brief Returns a pointer to contiguous read-only element storage.
	 * @return First element address, or `nullptr` when empty.
	 * @complexity O(1).
	 */
	[[nodiscard]] const_pointer data() const noexcept {return buffer_.data();}

	/** @brief Returns the first element. @throws Exceptions::IndexError If empty. @complexity O(1). */
	reference front() {return buffer_.front();}
	/** @brief Returns the first element. @throws Exceptions::IndexError If empty. @complexity O(1). */
	const_reference front() const {return buffer_.front();}

	/** @brief Returns the final element. @throws Exceptions::IndexError If empty. @complexity O(1). */
	reference back() {return buffer_.back();}
	/** @brief Returns the final element. @throws Exceptions::IndexError If empty. @complexity O(1). */
	const_reference back() const {return buffer_.back();}

	/** @brief Returns an element without bounds checking. @pre `index < size()`. @complexity O(1). */
	reference operator[](size_type index) noexcept {return buffer_[index];}
	/** @brief Returns an element without bounds checking. @pre `index < size()`. @complexity O(1). */
	const_reference operator[](size_type index) const noexcept {return buffer_[index];}

	/**
	 * @brief Returns an element using checked, Python-style flat indexing.
	 * @param index Index in the range `[-size(), size())`.
	 * @throws Exceptions::IndexError If @p index is outside the valid range.
	 * @complexity O(1).
	 */
	reference at(difference_type index) {return buffer_[indexing::normalize_index(index, size())];}
	/**
	 * @brief Returns an element using checked, Python-style flat indexing.
	 * @param index Index in the range `[-size(), size())`.
	 * @throws Exceptions::IndexError If @p index is outside the valid range.
	 * @complexity O(1).
	 */
	const_reference at(difference_type index) const {return buffer_[indexing::normalize_index(index, size())];}

	/** @brief Returns a mutable iterator to the first element. @complexity O(1). */
	iterator begin() noexcept {return buffer_.begin();}
	/** @brief Returns a const iterator to the first element. @complexity O(1). */
	const_iterator begin() const noexcept {return buffer_.begin();}
	/** @brief Returns a const iterator to the first element. @complexity O(1). */
	const_iterator cbegin() const noexcept {return buffer_.cbegin();}
	/** @brief Returns a mutable iterator past the final element. @complexity O(1). */
	iterator end() noexcept {return buffer_.end();}
	/** @brief Returns a const iterator past the final element. @complexity O(1). */
	const_iterator end() const noexcept {return buffer_.end();}
	/** @brief Returns a const iterator past the final element. @complexity O(1). */
	const_iterator cend() const noexcept {return buffer_.cend();}
	/** @brief Returns a mutable reverse iterator to the final element. @complexity O(1). */
	reverse_iterator rbegin() noexcept {return buffer_.rbegin();}
	/** @brief Returns a const reverse iterator to the final element. @complexity O(1). */
	const_reverse_iterator rbegin() const noexcept {return buffer_.rbegin();}
	/** @brief Returns a const reverse iterator to the final element. @complexity O(1). */
	const_reverse_iterator crbegin() const noexcept {return buffer_.crbegin();}
	/** @brief Returns the past-the-end mutable reverse iterator. @complexity O(1). */
	reverse_iterator rend() noexcept {return buffer_.rend();}
	/** @brief Returns the past-the-end const reverse iterator. @complexity O(1). */
	const_reverse_iterator rend() const noexcept {return buffer_.rend();}
	/** @brief Returns the past-the-end const reverse iterator. @complexity O(1). */
	const_reverse_iterator crend() const noexcept {return buffer_.crend();}

	/**
	 * @brief Assigns @p value to every stored element.
	 * @param value Value copied into each element.
	 * @throws Any exception thrown by value_type's copy-assignment operator.
	 * @complexity O(size()).
	 */
	void fill(const_reference value) {buffer_.fill(value);}

	/**
	 * @brief Exchanges storage and layout metadata with @p other.
	 *
	 * Pointers, references, and iterators remain valid but refer to elements now
	 * owned by the other object.
	 *
	 * @param other ArrayBase whose state is exchanged with this object.
	 * @complexity O(1).
	 */
	void swap(ArrayBase& other) noexcept
	{
		buffer_.swap(other.buffer_);
		shape_.swap(other.shape_);
		strides_.swap(other.strides_);
	}

protected:
	/**
	 * @brief Constructs value-initialized storage for @p shape.
	 * @param shape Logical shape of the resulting array.
	 * @throws Exceptions::DimensionError If the element count or a stride
	 *         cannot be represented by size_type.
	 * @throws std::bad_alloc If element or metadata allocation fails.
	 * @throws Any exception thrown while constructing a value_type element.
	 * @complexity O(shape.elements() + shape.rank()).
	 */
	explicit ArrayBase(const Shape& shape)
		: buffer_(shape.elements()),
		  shape_(shape),
		  strides_(shape)
	{}

	/**
	 * @brief Constructs storage filled with @p value for @p shape.
	 * @param shape Logical shape of the resulting array.
	 * @param value Value copied into every element.
	 * @throws Exceptions::DimensionError If the element count or a stride
	 *         cannot be represented by size_type.
	 * @throws std::bad_alloc If element or metadata allocation fails.
	 * @throws Any exception thrown by value_type's copy constructor.
	 * @complexity O(shape.elements() + shape.rank()).
	 */
	ArrayBase(const Shape& shape, const_reference value)
		: buffer_(shape.elements(), value),
		  shape_(shape),
		  strides_(shape)
	{}

	/**
	 * @brief Adopts an existing buffer for @p shape.
	 *
	 * Ownership of @p buffer is transferred before its element count is
	 * validated. If validation fails, the adopted storage is released while
	 * unwinding construction.
	 *
	 * @param shape Logical shape associated with the storage.
	 * @param buffer Element storage to adopt.
	 * @throws Exceptions::ShapeError If `buffer.size() != shape.elements()`.
	 * @throws Exceptions::DimensionError If the element count or a stride
	 *         cannot be represented by size_type.
	 * @throws std::bad_alloc If metadata allocation fails.
	 * @complexity O(shape.rank()) plus O(1) buffer ownership transfer.
	 */
	ArrayBase(const Shape& shape, Buffer<value_type>&& buffer)
		: buffer_(std::move(buffer)),
		  shape_(shape),
		  strides_(shape)
	{
		if (buffer_.size() != shape_.elements())
		{
			throw Exceptions::ShapeError(
				"Buffer size must match shape element count.");
		}
	}

	/**
	 * @brief Converts checked signed multidimensional indices to a flat offset.
	 *
	 * Each component is normalized independently against its corresponding
	 * dimension. Negative components count backward from the end. A custom
	 * component message replaces IndexError messages produced during component
	 * normalization; when it is null, the original exception is propagated.
	 *
	 * @tparam IndexContainer Sized, indexable container of signed indices.
	 * @param raw_indices One index per logical dimension.
	 * @param rank_mismatch_message Message used when the index count differs
	 *        from rank().
	 * @param component_oob_message Optional replacement message for an invalid
	 *        component.
	 * @return Row-major flat element offset.
	 * @throws Exceptions::IndexError If the rank differs or a component is out
	 *         of bounds.
	 * @complexity O(rank()).
	 */
	template<typename IndexContainer>
	size_type normalized_flat_offset(
		const IndexContainer& raw_indices,
		const char* rank_mismatch_message = "Multi-index rank must match array rank.",
		const char* component_oob_message = nullptr
	) const
	{
		if (raw_indices.size() != rank())
		{
			throw Exceptions::IndexError(rank_mismatch_message);
		}

		size_type offset = 0;

		for (size_type i = 0; i < rank(); ++i)
		{
			try
			{
				const size_type index =
					indexing::normalize_index(
						raw_indices[i],
						shape_[i]);

				offset += index * strides_[i];
			}
			catch (const Exceptions::IndexError&)
			{
				if (component_oob_message != nullptr)
				{
					throw Exceptions::IndexError(component_oob_message);
				}

				throw;
			}
		}

		return offset;
	}

	/**
	 * @brief Computes an unchecked row-major offset from normalized indices.
	 *
	 * @tparam IndexContainer Iterable container of non-negative indices.
	 * @param indices One already-normalized index per logical dimension.
	 * @return Row-major flat element offset.
	 * @pre `indices.size() == rank()` and every component is within its
	 *      corresponding dimension. Violating these requirements may produce an
	 *      invalid offset or undefined behavior.
	 * @complexity O(rank()).
	 */
	template<typename IndexContainer>
	size_type flat_offset(const IndexContainer& indices) const
	{
		return indexing::offset(shape_, strides_, indices);
	}

private:
	Buffer<value_type> buffer_;
	Shape shape_;
	Strides strides_;
};

}
