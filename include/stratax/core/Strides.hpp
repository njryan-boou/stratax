#pragma once

#include <cstddef>
#include <ostream>

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/indexing/Normalize.hpp>

namespace stratax::core {

class Strides
{
private:
	Buffer<std::size_t> buffer_;

public:
	using iterator = Buffer<std::size_t>::iterator;
	using const_iterator = Buffer<std::size_t>::const_iterator;
	using reverse_iterator = Buffer<std::size_t>::reverse_iterator;
	using const_reverse_iterator = Buffer<std::size_t>::const_reverse_iterator;

	Strides() noexcept = default;

	explicit Strides(const Shape& shape)
		: buffer_(shape.rank())
	{
		if (shape.empty()) {
			return;
		}

		buffer_[shape.rank() - 1] = 1;

		for (std::size_t i = shape.rank() - 1; i > 0; --i) {
			try
			{
				buffer_[i - 1] = validation::checked_multiply(
					buffer_[i],
					shape[i],
					"Strides overflow for shape");
			}
			catch (const Exceptions::DimensionError&)
			{
				buffer_[i - 1] = std::numeric_limits<std::size_t>::max();
			}
		}
	}

	[[nodiscard]] std::size_t* data() noexcept
	{
		return buffer_.data();
	}

	[[nodiscard]] const std::size_t* data() const noexcept
	{
		return buffer_.data();
	}

	[[nodiscard]] std::size_t& front()
	{
		if (empty()) {
			throw Exceptions::IndexError("Strides front cannot be accessed when the strides are empty.");
		}
		return buffer_.front();
	}

	[[nodiscard]] const std::size_t& front() const
	{
		if (empty()) {
			throw Exceptions::IndexError("Strides front cannot be accessed when the strides are empty.");
		}
		return buffer_.front();
	}

	[[nodiscard]] std::size_t& back()
	{
		if (empty()) {
			throw Exceptions::IndexError("Strides back cannot be accessed when the strides are empty.");
		}
		return buffer_.back();
	}

	[[nodiscard]] const std::size_t& back() const
	{
		if (empty()) {
			throw Exceptions::IndexError("Strides back cannot be accessed when the strides are empty.");
		}
		return buffer_.back();
	}

	[[nodiscard]] iterator begin() noexcept
	{
		return buffer_.begin();
	}

	[[nodiscard]] const_iterator begin() const noexcept
	{
		return buffer_.begin();
	}

	[[nodiscard]] const_iterator cbegin() const noexcept
	{
		return buffer_.cbegin();
	}

	[[nodiscard]] iterator end() noexcept
	{
		return buffer_.end();
	}

	[[nodiscard]] const_iterator end() const noexcept
	{
		return buffer_.end();
	}

	[[nodiscard]] const_iterator cend() const noexcept
	{
		return buffer_.cend();
	}

	[[nodiscard]] reverse_iterator rbegin() noexcept
	{
		return buffer_.rbegin();
	}

	[[nodiscard]] const_reverse_iterator rbegin() const noexcept
	{
		return buffer_.rbegin();
	}

	[[nodiscard]] const_reverse_iterator crbegin() const noexcept
	{
		return buffer_.crbegin();
	}

	[[nodiscard]] reverse_iterator rend() noexcept
	{
		return buffer_.rend();
	}

	[[nodiscard]] const_reverse_iterator rend() const noexcept
	{
		return buffer_.rend();
	}

	[[nodiscard]] const_reverse_iterator crend() const noexcept
	{
		return buffer_.crend();
	}

	[[nodiscard]] bool empty() const noexcept
	{
		return buffer_.empty();
	}

	[[nodiscard]] std::size_t rank() const noexcept
	{
		return buffer_.size();
	}

	const std::size_t& operator[](std::size_t index) const noexcept
	{
		return buffer_[index];
	}

	const std::size_t& at(std::ptrdiff_t index) const
{
    return buffer_[stratax::indexing::normalize_index(index, rank())];
}

	void swap(Strides& other) noexcept
	{
		buffer_.swap(other.buffer_);
	}

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

/** @brief Streams strides in tuple-like notation. */
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
