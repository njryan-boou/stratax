#pragma once

#include <cstddef>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/validation/Validation.hpp>

namespace stratax::core {

/** @brief Stores strides for a shape in contiguous memory. */
class Strides
{
private:
	Buffer<std::size_t> buffer_;

public:
	/** @brief Iterator over immutable stride values. */
	using iterator = Buffer<std::size_t>::const_iterator;

	/** @brief Const iterator over stride values. */
	using const_iterator = Buffer<std::size_t>::const_iterator;

	/** @brief Reverse iterator over immutable stride values. */
	using reverse_iterator = Buffer<std::size_t>::const_reverse_iterator;

	/** @brief Const reverse iterator over stride values. */
	using const_reverse_iterator = Buffer<std::size_t>::const_reverse_iterator;

	/** @brief Creates an empty stride vector. */
	Strides() noexcept = default;

	/** @brief Creates row-major strides for a given shape. */
	explicit Strides(const Shape& shape)
		: buffer_(shape.rank())
	{
		if (shape.empty()) {
			return;
		}

		buffer_[shape.rank() - 1] = 1;

		for (std::size_t i = shape.rank() - 1; i > 0; --i) {
			// Preserve constructibility for extremely large shapes by
			// saturating intermediate stride values when they overflow.
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

	/** @brief Copies stride metadata. */
	Strides(const Strides&) = default;

	/** @brief Moves stride metadata. */
	Strides(Strides&&) noexcept = default;

	/** @brief Copy-assigns stride metadata. */
	Strides& operator=(const Strides&) = default;

	/** @brief Move-assigns stride metadata. */
	Strides& operator=(Strides&&) noexcept = default;

	/** @brief Destroys the stride vector. */
	~Strides() = default;

	 /** @brief Returns the number of stored stride values. */
	[[nodiscard]] std::size_t size() const noexcept
	{
		return buffer_.size();
	}

	 /** @brief Returns whether the stride vector is empty. */
	[[nodiscard]] bool empty() const noexcept
	{
		return buffer_.empty();
	}

	 /** @brief Returns the number of dimensions represented by the strides. */
	[[nodiscard]] std::size_t rank() const noexcept
	{
		return buffer_.size();
	}

	/** @brief Returns a stride value without bounds checking. */
	const std::size_t& operator[](std::size_t index) const noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns a stride value without bounds checking. */
	const std::size_t& operator()(std::size_t index) const noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns a stride value with bounds checking. */
	const std::size_t& at(std::ptrdiff_t index) const
	{
		const std::size_t normalized = validation::normalize_index(index, rank(), "Strides index out of bounds");
		return buffer_[normalized];
	}

	 /** @brief Returns the first stride value. */
	const std::size_t& front() const
	{
		return buffer_.front();
	}

	 /** @brief Returns the last stride value. */
	const std::size_t& back() const
	{
		return buffer_.back();
	}

	 /** @brief Returns the raw stride data pointer. */
	[[nodiscard]] const std::size_t* data() const noexcept
	{
		return buffer_.data();
	}

	 /** @brief Returns an iterator to the first stride value. */
	[[nodiscard]] iterator begin() noexcept
	{
		return buffer_.begin();
	}

	 /** @brief Returns a const iterator to the first stride value. */
	[[nodiscard]] const_iterator begin() const noexcept
	{
		return buffer_.begin();
	}

	 /** @brief Returns a const iterator to the first stride value. */
	[[nodiscard]] const_iterator cbegin() const noexcept
	{
		return buffer_.cbegin();
	}

	 /** @brief Returns an iterator one past the last stride value. */
	[[nodiscard]] iterator end() noexcept
	{
		return buffer_.end();
	}

	 /** @brief Returns a const iterator one past the last stride value. */
	[[nodiscard]] const_iterator end() const noexcept
	{
		return buffer_.end();
	}

	 /** @brief Returns a const iterator one past the last stride value. */
	[[nodiscard]] const_iterator cend() const noexcept
	{
		return buffer_.cend();
	}

	 /** @brief Returns a reverse iterator to the last stride value. */
	[[nodiscard]] reverse_iterator rbegin() noexcept
	{
		return buffer_.rbegin();
	}

	 /** @brief Returns a const reverse iterator to the last stride value. */
	[[nodiscard]] const_reverse_iterator rbegin() const noexcept
	{
		return buffer_.rbegin();
	}

	 /** @brief Returns a const reverse iterator to the last stride value. */
	[[nodiscard]] const_reverse_iterator crbegin() const noexcept
	{
		return buffer_.crbegin();
	}

	 /** @brief Returns a reverse iterator before the first stride value. */
	[[nodiscard]] reverse_iterator rend() noexcept
	{
		return buffer_.rend();
	}

	 /** @brief Returns a const reverse iterator before the first stride value. */
	[[nodiscard]] const_reverse_iterator rend() const noexcept
	{
		return buffer_.rend();
	}

	 /** @brief Returns a const reverse iterator before the first stride value. */
	[[nodiscard]] const_reverse_iterator crend() const noexcept
	{
		return buffer_.crend();
	}

	 /** @brief Swaps the stored strides with another instance. */
	void swap(Strides& other) noexcept
	{
		buffer_.swap(other.buffer_);
	}

	 /** @brief Compares two stride vectors for equality. */
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

	/** @brief Returns whether two stride vectors differ. */
	[[nodiscard]] bool operator!=(const Strides& other) const noexcept
	{
		return !(*this == other);
	}
};

/** @brief Streams strides in tuple-like notation. */
inline std::ostream& operator<<(std::ostream& os, const Strides& stride)
{
	os << "(";

	bool first = true;
	for (std::size_t dim : stride)
	{
		if (!first)
			os << ", ";

		os << dim;
		first = false;
	}

	if (stride.rank() == 1)
	{
		os << ",";
	}

	os << ")";

	return os;
}

}
