#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <stratax/core/Shape.hpp>
#include <stratax/indexing/Indexing.hpp>

namespace stratax::core {

template<typename T>
class ArrayView
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

	ArrayView(pointer data, const Shape& shape, const Shape& strides)
		: data_(data),
		  shape_(shape),
		  strides_(strides)
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
        return shape_.rank();
    }

    [[nodiscard]]
    size_type size() const noexcept
    {
        return shape_.elements();
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

	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	reference operator()(size_type first, Rest... rest)
	{
		const std::array<size_type, sizeof...(Rest) + 1> indices{
			first,
			static_cast<size_type>(rest)...
		};

		return data_[indexing::offset(strides_, indices)];
	}

	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	const_reference operator()(size_type first, Rest... rest) const
	{
		const std::array<size_type, sizeof...(Rest) + 1> indices{
			first,
			static_cast<size_type>(rest)...
		};

		return data_[indexing::offset(strides_, indices)];
	}

private:
    pointer data_;
    Shape shape_;
    Shape strides_;
};

} // namespace stratax::core
