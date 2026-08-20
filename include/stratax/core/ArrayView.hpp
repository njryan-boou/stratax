#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include <stratax/core/Shape.hpp>
#include <stratax/indexing/Indexing.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::core {

template<typename T>
class ArrayView
{
public:
	using element_type = T;
    using value_type = std::remove_const_t<element_type>;
    using size_type = std::size_t;
	using reference = element_type&;
	using const_reference = const value_type&;
	using pointer = element_type*;
	using const_pointer = const value_type*;
	using const_view = ArrayView<const value_type>;

	template<bool IsConst>
	class basic_iterator
	{
	private:
		using view_type = std::conditional_t<IsConst, const ArrayView, ArrayView>;

		view_type* view_ = nullptr;
		size_type index_ = 0;

	public:
		using iterator_concept = std::random_access_iterator_tag;
		using iterator_category = std::random_access_iterator_tag;
		using value_type = ArrayView::value_type;
		using difference_type = std::ptrdiff_t;
		using reference = std::conditional_t<IsConst, const_reference, ArrayView::reference>;

		basic_iterator() = default;
		basic_iterator(view_type* view, size_type index) noexcept
			: view_(view), index_(index)
		{}

		template<bool OtherConst>
		requires (IsConst && !OtherConst)
		basic_iterator(const basic_iterator<OtherConst>& other) noexcept
			: view_(other.view_), index_(other.index_)
		{}

		reference operator*() const {return (*view_)[index_];}
		reference operator[](difference_type offset) const {return *(*this + offset);}

		basic_iterator& operator++() {++index_; return *this;}
		basic_iterator operator++(int) {auto copy = *this; ++*this; return copy;}
		basic_iterator& operator--() {--index_; return *this;}
		basic_iterator operator--(int) {auto copy = *this; --*this; return copy;}

		basic_iterator& operator+=(difference_type offset)
		{
			index_ = static_cast<size_type>(static_cast<difference_type>(index_) + offset);
			return *this;
		}

		basic_iterator& operator-=(difference_type offset) {return *this += -offset;}

		friend basic_iterator operator+(basic_iterator iterator, difference_type offset)
		{
			iterator += offset;
			return iterator;
		}

		friend basic_iterator operator+(difference_type offset, basic_iterator iterator)
		{
			return iterator + offset;
		}

		friend basic_iterator operator-(basic_iterator iterator, difference_type offset)
		{
			iterator -= offset;
			return iterator;
		}

		friend difference_type operator-(const basic_iterator& lhs, const basic_iterator& rhs)
		{
			return static_cast<difference_type>(lhs.index_) -
				static_cast<difference_type>(rhs.index_);
		}

		friend bool operator==(const basic_iterator&, const basic_iterator&) = default;
		friend auto operator<=>(const basic_iterator& lhs, const basic_iterator& rhs)
		{
			return lhs.index_ <=> rhs.index_;
		}

		template<bool>
		friend class basic_iterator;
	};

	using iterator = basic_iterator<false>;
	using const_iterator = basic_iterator<true>;

	ArrayView(pointer data, const Shape& shape, const Shape& strides)
		: data_(data),
		  shape_(shape),
		  strides_(strides)
	{}

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

    [[nodiscard]]
    const Shape& shape() const noexcept
    {
        return shape_;
    }

    [[nodiscard]]
    const Shape& strides() const noexcept
    {
        return strides_;
    }

    [[nodiscard]]
    size_type ndim() const noexcept
    {
		return rank();
    }

	[[nodiscard]]
	size_type rank() const noexcept
	{
		return shape_.rank();
	}

    [[nodiscard]]
    size_type size() const noexcept
    {
        return shape_.elements();
    }

	[[nodiscard]]
	bool empty() const noexcept
	{
		return size() == 0;
	}

    [[nodiscard]]
    pointer data() noexcept
    {
        return data_;
    }

    [[nodiscard]]
    const_pointer data() const noexcept
    {
        return data_;
    }

	[[nodiscard]]
	const_view as_const() const
	{
		return const_view(data_, shape_, strides_);
	}

	reference operator[](size_type index)
	{
		return data_[flat_offset(index)];
	}

	const_reference operator[](size_type index) const
	{
		return data_[flat_offset(index)];
	}

	iterator begin() noexcept {return iterator(this, 0);}
	const_iterator begin() const noexcept {return const_iterator(this, 0);}
	const_iterator cbegin() const noexcept {return const_iterator(this, 0);}
	iterator end() noexcept {return iterator(this, size());}
	const_iterator end() const noexcept {return const_iterator(this, size());}
	const_iterator cend() const noexcept {return const_iterator(this, size());}

	template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
reference operator()(size_type first, Rest... rest)
{
    constexpr size_type count = sizeof...(Rest) + 1;

    if (count != rank())
    {
        throw Exceptions::IndexError(
            "Number of indices must match view rank.");
    }

    const std::array<size_type, count> indices{
        first,
        static_cast<size_type>(rest)...
    };

    return data_[indexing::offset(strides_, indices)];
}

template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
const_reference operator()(size_type first, Rest... rest) const
{
    constexpr size_type count = sizeof...(Rest) + 1;

    if (count != rank())
    {
        throw Exceptions::IndexError(
            "Number of indices must match view rank.");
    }

    const std::array<size_type, count> indices{
        first,
        static_cast<size_type>(rest)...
    };

    return data_[indexing::offset(strides_, indices)];
}

reference at(difference_type index)
{
    index = indexing::normalize_index(index, size());
    return (*this)[static_cast<size_type>(index)];
}

const_reference at(difference_type index) const
{
    index = indexing::normalize_index(index, size());
    return (*this)[static_cast<size_type>(index)];
}

reference at(const std::vector<difference_type>& indices)
{
    return data_[checked_offset(indices)];
}

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
        throw Exceptions::IndexError(
            "ArrayView index rank must match view rank.");
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
