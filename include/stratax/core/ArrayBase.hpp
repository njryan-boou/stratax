#pragma once

#include <cstddef>
#include <utility>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Indexing.hpp>
#include <stratax/indexing/Normalize.hpp>

namespace stratax::core {

template<typename T>
class ArrayBase
{
public:
	using value_type = T;
	using iterator = typename Buffer<T>::iterator;
	using const_iterator = typename Buffer<T>::const_iterator;
	using reverse_iterator = typename Buffer<T>::reverse_iterator;
	using const_reverse_iterator = typename Buffer<T>::const_reverse_iterator;

	[[nodiscard]] std::size_t size() const noexcept
	{
		return buffer_.size();
	}

	[[nodiscard]] bool empty() const noexcept
	{
		return buffer_.empty();
	}

	[[nodiscard]] std::size_t rank() const noexcept
	{
		return shape_.rank();
	}

	[[nodiscard]] const Shape& shape() const noexcept
	{
		return shape_;
	}

	[[nodiscard]] const Strides& strides() const noexcept
	{
		return strides_;
	}

	[[nodiscard]] T* data() noexcept
	{
		return buffer_.data();
	}

	[[nodiscard]] const T* data() const noexcept
	{
		return buffer_.data();
	}

	T& front()
	{
		return buffer_.front();
	}

	const T& front() const
	{
		return buffer_.front();
	}

	T& back()
	{
		return buffer_.back();
	}

	const T& back() const
	{
		return buffer_.back();
	}

	T& operator[](std::size_t index) noexcept
	{
		return buffer_[index];
	}

	const T& operator[](std::size_t index) const noexcept
	{
		return buffer_[index];
	}

	T& at(std::ptrdiff_t index)
	{
		return buffer_[stratax::indexing::normalize_index(index, size())];
	}

	const T& at(std::ptrdiff_t index) const
	{
		return buffer_[stratax::indexing::normalize_index(index, size())];
	}

	iterator begin() noexcept
	{
		return buffer_.begin();
	}

	const_iterator begin() const noexcept
	{
		return buffer_.begin();
	}

	const_iterator cbegin() const noexcept
	{
		return buffer_.cbegin();
	}

	iterator end() noexcept
	{
		return buffer_.end();
	}

	const_iterator end() const noexcept
	{
		return buffer_.end();
	}

	const_iterator cend() const noexcept
	{
		return buffer_.cend();
	}

	reverse_iterator rbegin() noexcept
	{
		return buffer_.rbegin();
	}

	const_reverse_iterator rbegin() const noexcept
	{
		return buffer_.rbegin();
	}

	const_reverse_iterator crbegin() const noexcept
	{
		return buffer_.crbegin();
	}

	reverse_iterator rend() noexcept
	{
		return buffer_.rend();
	}

	const_reverse_iterator rend() const noexcept
	{
		return buffer_.rend();
	}

	const_reverse_iterator crend() const noexcept
	{
		return buffer_.crend();
	}

	void fill(const T& value)
	{
		buffer_.fill(value);
	}

protected:
	explicit ArrayBase(const Shape& shape)
		: buffer_(shape.elements()),
		  shape_(shape),
		  strides_(shape)
	{}

	ArrayBase(const Shape& shape, const T& value)
		: buffer_(shape.elements(), value),
		  shape_(shape),
		  strides_(shape)
	{}

	ArrayBase(const Shape& shape, Buffer<T>&& buffer)
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

	template<typename IndexContainer>
	std::size_t normalized_flat_offset(
		const IndexContainer& raw_indices,
		const char* rank_mismatch_message = "Multi-index rank must match array rank.",
		const char* component_oob_message = nullptr) const
	{
		if (raw_indices.size() != rank())
		{
			throw Exceptions::IndexError(rank_mismatch_message);
		}

		std::size_t offset = 0;

		for (std::size_t i = 0; i < rank(); ++i)
		{
			try
			{
				const std::size_t index =
					stratax::indexing::normalize_index(
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

	template<typename IndexContainer>
	std::size_t flat_offset(const IndexContainer& indices) const
	{
		return stratax::indexing::offset(shape_, strides_, indices);
	}

	Buffer<T> buffer_;
	Shape shape_;
	Strides strides_;
};

}
