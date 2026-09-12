/** @file
 * @brief Non-owning views and logical random-access iterators.
 */
#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include <stratax/core/Shape.hpp>
#include <stratax/indexing/Indexing.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::core {

/**
 * @brief Non-owning view of elements with unsigned element strides.
 *
 * The view copies shape/stride metadata but never owns or extends the lifetime
 * of the referenced storage. Copies alias the same elements. The caller must
 * keep that allocation alive and large enough for every reachable offset.
 * Strides may describe transposes, gaps, or repeated elements (zero strides);
 * negative strides are not representable. Flat access follows logical row-major
 * order, not necessarily physical storage order.
 *
 * @tparam T Element type, optionally const; const elements prohibit writes.
 * @invariant Shape and stride ranks agree and the element count is representable.
 * @invariant Nonempty views have non-null data and representable maximum offsets.
 * @note Const access to the view returns const element references. Constness
 *       does not prevent another alias from modifying the underlying storage.
 */
template<typename T>
class ArrayView
{
public:
	/** @brief Element type including any const qualifier. */
	using element_type = T;
    /** @brief Unqualified element type. */
    using value_type = std::remove_const_t<element_type>;
    /** @brief Unsigned extent and offset type. */
    using size_type = std::size_t;
	/** @brief Signed index and iterator-distance type. */
	using difference_type = std::ptrdiff_t;
	/** @brief Element reference, mutable only when T is mutable. */
	using reference = element_type&;
	/** @brief Read-only element reference. */
	using const_reference = const value_type&;
	/** @brief Element pointer with the constness of T. */
	using pointer = element_type*;
	/** @brief Read-only element pointer. */
	using const_pointer = const value_type*;
	/** @brief View type exposing read-only elements. */
	using const_view = ArrayView<const value_type>;

	/**
	 * @brief Random-access iterator over logical flat indices of a view.
	 * @tparam IsConst Whether dereferencing provides read-only access.
	 * @pre The referenced view object and its underlying allocation remain alive
	 *      and unchanged in layout while the iterator is used.
	 * @pre Dereference selects an index below size(); arithmetic stays within
	 *      [begin(), end()], and ordering/difference compare the same view.
	 * @complexity Dereference and subscripting are O(view.rank()); arithmetic
	 *             and comparison are O(1). The iterator is not contiguous.
	 */
	template<bool IsConst>
	class basic_iterator
	{
	private:
		using view_type = std::conditional_t<IsConst, const ArrayView, ArrayView>;

		view_type* view_ = nullptr;
		size_type index_ = 0;

	public:
		/** @brief C++20 random-access iterator concept tag. */
		using iterator_concept = std::random_access_iterator_tag;
		/** @brief Legacy random-access iterator category tag. */
		using iterator_category = std::random_access_iterator_tag;
		/** @brief Unqualified element type. */
		using value_type = ArrayView::value_type;
		/** @brief Signed iterator-distance type. */
		using difference_type = std::ptrdiff_t;
		/** @brief Element reference with the iterator constness. */
		using reference = std::conditional_t<IsConst, const_reference, ArrayView::reference>;

		/** @brief Constructs a singular iterator. */
		basic_iterator() = default;
		/** @brief Constructs an iterator at a logical flat index, including the end position. */
		basic_iterator(view_type* view, size_type index) noexcept
			: view_(view), index_(index)
		{}

		/** @brief Converts a mutable iterator to a read-only iterator. */
		template<bool OtherConst>
		requires (IsConst && !OtherConst)
		basic_iterator(const basic_iterator<OtherConst>& other) noexcept
			: view_(other.view_), index_(other.index_)
		{}

		/** @brief Returns the element at the current logical position. */
		reference operator*() const {return (*view_)[index_];}
		/** @brief Returns the element at a relative logical offset. */
		reference operator[](difference_type offset) const {return *(*this + offset);}

		/** @brief Advances one position and returns this iterator. */
		basic_iterator& operator++() {++index_; return *this;}
		/** @brief Advances one position and returns the previous iterator. */
		basic_iterator operator++(int) {auto copy = *this; ++*this; return copy;}
		/** @brief Moves back one position and returns this iterator. */
		basic_iterator& operator--() {--index_; return *this;}
		/** @brief Moves back one position and returns the previous iterator. */
		basic_iterator operator--(int) {auto copy = *this; --*this; return copy;}

		/** @brief Advances by a signed offset and returns this iterator. */
		basic_iterator& operator+=(difference_type offset)
		{
			index_ = static_cast<size_type>(static_cast<difference_type>(index_) + offset);
			return *this;
		}

		/** @brief Moves back by a signed offset and returns this iterator. */
		basic_iterator& operator-=(difference_type offset) {return *this += -offset;}

		/** @brief Returns an iterator advanced by a signed offset. */
		friend basic_iterator operator+(basic_iterator iterator, difference_type offset)
		{
			iterator += offset;
			return iterator;
		}

		/** @brief Returns an iterator advanced by a signed offset. */
		friend basic_iterator operator+(difference_type offset, basic_iterator iterator)
		{
			return iterator + offset;
		}

		/** @brief Returns an iterator moved back by a signed offset. */
		friend basic_iterator operator-(basic_iterator iterator, difference_type offset)
		{
			iterator -= offset;
			return iterator;
		}

		/** @brief Returns the signed logical distance between iterators into the same view. */
		friend difference_type operator-(const basic_iterator& lhs, const basic_iterator& rhs)
		{
			return static_cast<difference_type>(lhs.index_) -
				static_cast<difference_type>(rhs.index_);
		}

		/** @brief Compares both the view identity and logical position. */
		friend bool operator==(const basic_iterator&, const basic_iterator&) = default;
		/** @brief Orders logical positions within the same view. */
		friend auto operator<=>(const basic_iterator& lhs, const basic_iterator& rhs)
		{
			return lhs.index_ <=> rhs.index_;
		}

		template<bool>
		friend class basic_iterator;
	};
	/** @brief Mutable-view random-access iterator in logical order. */
	using iterator = basic_iterator<false>;
	/** @brief Read-only random-access iterator in logical order. */
	using const_iterator = basic_iterator<true>;

	/**
	 * @brief Constructs a view after validating shape and stride metadata.
	 * @param data Base address; may be null only for an empty view.
	 * @param shape Logical extents; rank zero describes an empty view.
	 * @param strides Non-negative strides measured in elements, one per axis.
	 * @pre The caller provides live storage covering every reachable element.
	 * @throws Exceptions::RankError If shape and stride ranks differ.
	 * @throws Exceptions::ValueError If nonempty storage has a null pointer.
	 * @throws Exceptions::DimensionError If the element count or maximum offset overflows.
	 * @throws std::bad_alloc If metadata allocation fails.
	 * @complexity O(shape.rank()).
	 */
	ArrayView(pointer data, const Shape& shape, const Shape& strides)
		: data_(data),
		  shape_(shape),
		  strides_(strides)
	{
		if (shape_.rank() != strides_.rank())
		{
			throw Exceptions::RankError("View shape and strides must have the same rank.");
		}
		if (shape_.elements() == 0) return;
		if (data_ == nullptr)
		{
			throw Exceptions::ValueError("A nonempty view requires storage.");
		}
		size_type maximum_offset = 0;
		for (size_type dim = 0; dim < rank(); ++dim)
		{
			const auto extent = shape_[dim] - 1;
			const auto limit = std::numeric_limits<size_type>::max();
			if (extent != 0 && strides_[dim] > (limit - maximum_offset) / extent)
			{
				throw Exceptions::DimensionError("View offset overflow.");
			}
			maximum_offset += extent * strides_[dim];
		}
	}

	/**
	 * @brief Creates a read-only alias of a mutable view, copying its metadata.
	 * @param other Source view; its allocation lifetime is still managed externally.
	 * @throws std::bad_alloc If copying metadata fails.
	 * @complexity O(other.rank()).
	 */
	template<typename U>
	requires (
		std::is_const_v<element_type> &&
		!std::is_const_v<U> &&
		std::same_as<value_type, std::remove_const_t<U>>
	)
	ArrayView(const ArrayView<U>& other)
		: data_(other.data()),
		  shape_(other.shape()),
		  strides_(other.strides())
	{}

    /** @brief Logical extents. @complexity O(1). */
    [[nodiscard]]
    const Shape& shape() const noexcept
    {
        return shape_;
    }

    /** @brief Stored element strides. @complexity O(1). */
    [[nodiscard]]
    const Shape& strides() const noexcept
    {
        return strides_;
    }

    /** @brief Alias for rank(). @complexity O(1). */
    [[nodiscard]]
    size_type ndim() const noexcept
    {
		return rank();
    }

	/** @brief Number of logical axes. @complexity O(1). */
	[[nodiscard]]
	size_type rank() const noexcept
	{
		return shape_.rank();
	}

    /** @brief Number of logical elements, zero for rank zero. @complexity O(rank()). */
    [[nodiscard]]
    size_type size() const noexcept
    {
        return shape_.elements();
    }

	/** @brief Whether there are no logical elements. @complexity O(rank()). */
	[[nodiscard]]
	bool empty() const noexcept
	{
		return size() == 0;
	}

    /** @brief Base storage pointer; empty views may have non-null pointers. @complexity O(1). */
    [[nodiscard]]
    pointer data() noexcept
    {
        return data_;
    }

    /** @brief Read-only base pointer; empty views may have non-null pointers. @complexity O(1). */
    [[nodiscard]]
    const_pointer data() const noexcept
    {
        return data_;
    }

	/** @brief Read-only alias with copied metadata. @throws std::bad_alloc If allocation fails. @complexity O(rank()). */
	[[nodiscard]]
	const_view as_const() const
	{
		return const_view(data_, shape_, strides_);
	}

	/** @brief Unchecked logical flat access. @pre index < size(). @complexity O(rank()). */
	reference operator[](size_type index)
	{
		return data_[flat_offset(index)];
	}

	/** @brief Unchecked logical flat access. @pre index < size(). @complexity O(rank()). */
	const_reference operator[](size_type index) const
	{
		return data_[flat_offset(index)];
	}

	/** @brief First logical-element iterator. @complexity O(1). */
	iterator begin() noexcept {return iterator(this, 0);}
	/** @brief First logical-element iterator. @complexity O(1). */
	const_iterator begin() const noexcept {return const_iterator(this, 0);}
	/** @brief First logical-element iterator. @complexity O(1). */
	const_iterator cbegin() const noexcept {return const_iterator(this, 0);}
	/** @brief Past-the-end iterator. @complexity O(rank()). */
	iterator end() noexcept {return iterator(this, size());}
	/** @brief Past-the-end iterator. @complexity O(rank()). */
	const_iterator end() const noexcept {return const_iterator(this, size());}
	/** @brief Past-the-end iterator. @complexity O(rank()). */
	const_iterator cend() const noexcept {return const_iterator(this, size());}

	/**
 * @brief Accesses an element using multidimensional indices with rank checking only.
 * @param first First non-negative axis index.
 * @param rest Remaining non-negative axis indices.
 * @pre Every component is within its axis extent; bounds are not checked.
 * @throws Exceptions::RankError If the number of components differs from rank().
 * @complexity O(rank()).
 */
template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
reference operator()(size_type first, Rest... rest)
{
    constexpr size_type count = sizeof...(Rest) + 1;

    if (count != rank())
    {
        throw Exceptions::RankError(
            "The number of indices must match the view rank.");
    }

    const std::array<size_type, count> indices{
        first,
        static_cast<size_type>(rest)...
    };

    return data_[indexing::offset(strides_, indices)];
}

/**
 * @brief Accesses an element using multidimensional indices with rank checking only.
 * @param first First non-negative axis index.
 * @param rest Remaining non-negative axis indices.
 * @pre Every component is within its axis extent; bounds are not checked.
 * @throws Exceptions::RankError If the number of components differs from rank().
 * @complexity O(rank()).
 */
template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
const_reference operator()(size_type first, Rest... rest) const
{
    constexpr size_type count = sizeof...(Rest) + 1;

    if (count != rank())
    {
        throw Exceptions::RankError(
            "The number of indices must match the view rank.");
    }

    const std::array<size_type, count> indices{
        first,
        static_cast<size_type>(rest)...
    };

    return data_[indexing::offset(strides_, indices)];
}

/** @brief Checked logical flat access with negative indices.
 * @param index Signed flat index; -1 selects the last logical element.
 * @throws Exceptions::IndexError If out of bounds, including every index of an empty view.
 * @complexity O(rank()). */
reference at(difference_type index)
{
    index = indexing::normalize_index(index, size());
    return (*this)[static_cast<size_type>(index)];
}

/** @brief Checked logical flat access with negative indices.
 * @param index Signed flat index; -1 selects the last logical element.
 * @throws Exceptions::IndexError If out of bounds, including every index of an empty view.
 * @complexity O(rank()). */
const_reference at(difference_type index) const
{
    index = indexing::normalize_index(index, size());
    return (*this)[static_cast<size_type>(index)];
}

/** @brief Checked multidimensional access with negative indices.
 * @param indices One signed component per axis.
 * @pre The view has positive rank; rank-zero views contain no element.
 * @throws Exceptions::RankError If the component count differs from rank().
 * @throws Exceptions::IndexError If a component is out of bounds.
 * @complexity O(rank()). */
reference at(const std::vector<difference_type>& indices)
{
    return data_[checked_offset(indices)];
}

/** @brief Checked multidimensional access with negative indices.
 * @param indices One signed component per axis.
 * @pre The view has positive rank; rank-zero views contain no element.
 * @throws Exceptions::RankError If the component count differs from rank().
 * @throws Exceptions::IndexError If a component is out of bounds.
 * @complexity O(rank()). */
const_reference at(const std::vector<difference_type>& indices) const
{
    return data_[checked_offset(indices)];
}

private:
	[[nodiscard]]
	size_type flat_offset(size_type index) const
	{
		size_type offset = 0;

		for (size_type dim = rank(); dim-- > 0;)
		{
			const size_type coordinate = index % shape_[dim];
			index /= shape_[dim];
			offset += coordinate * strides_[dim];
		}

		return offset;
	}

	size_type checked_offset(
    const std::vector<difference_type>& indices) const
{
    if (indices.size() != rank())
    {
        throw Exceptions::RankError(
            "The number of indices must match the view rank.");
    }

    size_type offset = 0;

    for (size_type dim = 0; dim < rank(); ++dim)
    {
        const difference_type index =
            indexing::normalize_index(
                indices[dim],
                shape_[dim]);

        offset +=
            static_cast<size_type>(index) *
            strides_[dim];
    }

    return offset;
}

    pointer data_;
    Shape shape_;
    Shape strides_;
};

} // namespace stratax::core
