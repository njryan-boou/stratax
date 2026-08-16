#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <vector>

#include <stratax/core/Buffer.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Normalize.hpp>

namespace stratax::core {

class Shape
{
private:
	Buffer<std::size_t> dims_;

public:
	using const_iterator = Buffer<std::size_t>::const_iterator;
	using const_reverse_iterator = Buffer<std::size_t>::const_reverse_iterator;

	Shape() noexcept = default;

	Shape(std::initializer_list<std::size_t> dims)
		: dims_(dims)
	{}

	Shape(const std::vector<std::size_t>& dims)
		: dims_(dims.size())
	{
		std::copy(dims.begin(), dims.end(), dims_.begin());
	}

	[[nodiscard]] std::size_t elements() const
	{
		if (empty())
		{
			return 0;
		}
		std::size_t prod = 1;
		for (std::size_t dim : dims_)
		{
			if (prod > std::numeric_limits<std::size_t>::max() / dim)
			{
				throw Exceptions::DimensionError(
					"Shape has too many elements to fit in a size_t.");
			}
			prod *= dim;
		}
		return prod;
	}

	[[nodiscard]] std::size_t rank() const noexcept
	{
		return dims_.size();
	}

	[[nodiscard]] const std::size_t& operator[](std::size_t index) const noexcept
	{
		return dims_[index];
	}

	const std::size_t& at(std::ptrdiff_t index) const
	{
		return dims_[stratax::indexing::normalize_index(index, rank())];
	}

	[[nodiscard]] bool empty() const noexcept
	{
		return dims_.empty();
	}

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

	[[nodiscard]] const_iterator begin() const noexcept
	{
		return dims_.begin();
	}

	[[nodiscard]] const_iterator end() const noexcept
	{
		return dims_.end();
	}

	[[nodiscard]] const_iterator cbegin() const noexcept
	{
		return dims_.cbegin();
	}

	[[nodiscard]] const_iterator cend() const noexcept
	{
		return dims_.cend();
	}

	[[nodiscard]] const_reverse_iterator rbegin() const noexcept
	{
		return dims_.rbegin();
	}

	[[nodiscard]] const_reverse_iterator crbegin() const noexcept
	{
		return dims_.crbegin();
	}

	[[nodiscard]] const_reverse_iterator rend() const noexcept
	{
		return dims_.rend();
	}

	[[nodiscard]] const_reverse_iterator crend() const noexcept
	{
		return dims_.crend();
	}

	void swap(Shape& other) noexcept
	{
		dims_.swap(other.dims_);
	}

};

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
