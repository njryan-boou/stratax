#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/indexing/Indexing.hpp>

namespace stratax::container {

/** @brief Stores an N-dimensional Stratax array in contiguous memory. */
template<typename T>
requires Numeric<T>
class Tensor : public stratax::core::ArrayBase<T>
{
public:
	/** @brief Element type stored by the tensor. */
	using value_type = T;

	/** @brief Mutable iterator over tensor elements. */
	using iterator = typename core::Buffer<T>::iterator;

	/** @brief Const iterator over tensor elements. */
	using const_iterator = typename core::Buffer<T>::const_iterator;

	/** @brief Mutable reverse iterator over tensor elements. */
	using reverse_iterator = typename core::Buffer<T>::reverse_iterator;

	/** @brief Const reverse iterator over tensor elements. */
	using const_reverse_iterator = typename core::Buffer<T>::const_reverse_iterator;

	/** @brief Rebinds the tensor container to another element type. */
	template<typename U>
	using rebind = Tensor<U>;

	/** @brief Creates an empty tensor with rank 0. */
	Tensor() noexcept = default;

	/** @brief Creates a tensor with default-initialized storage for a shape. */
	explicit Tensor(const core::Shape& shape)
		: shape_(shape),
		  strides_(shape_),
		  buffer_(shape_.elements())
	{
	}

	/** @brief Creates a tensor from a shape and fills it with a value. */
	Tensor(const core::Shape& shape, const T& value)
		: shape_(shape),
		  strides_(shape_),
		  buffer_(shape_.elements(), value)
	{
	}

	/** @brief Creates a copy of another tensor. */
	Tensor(const Tensor&) = default;

	/** @brief Transfers ownership from another tensor. */
	Tensor(Tensor&&) noexcept = default;

	/** @brief Replaces this tensor with a copy of another tensor. */
	Tensor& operator=(const Tensor&) = default;

	/** @brief Replaces this tensor by taking ownership from another tensor. */
	Tensor& operator=(Tensor&&) noexcept = default;

	/** @brief Destroys the tensor. */
	~Tensor() = default;

	/** @brief Returns the total number of elements in the tensor. */
	[[nodiscard]] std::size_t size() const noexcept
	{
		return buffer_.size();
	}

	/** @brief Returns whether the tensor contains no elements. */
	[[nodiscard]] bool empty() const noexcept
	{
		return buffer_.empty();
	}

	/** @brief Returns the tensor rank. */
	[[nodiscard]] std::size_t rank() const noexcept
	{
		return shape_.rank();
	}

	/** @brief Returns the tensor shape. */
	const core::Shape& shape() const noexcept
	{
		return shape_;
	}

	/** @brief Returns the tensor strides. */
	const core::Strides& strides() const noexcept
	{
		return strides_;
	}

	/** @brief Returns a flat element without bounds checking. */
	T& operator()(std::size_t index) noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns a flat element without bounds checking. */
	const T& operator()(std::size_t index) const noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns a flat element without bounds checking. */
	T& operator[](std::size_t index) noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns a flat element without bounds checking. */
	const T& operator[](std::size_t index) const noexcept
	{
		return buffer_[index];
	}

	/** @brief Returns an element by multidimensional index with bounds checking. */
	template<typename... Rest>
	T& operator()(std::size_t first, std::size_t second, Rest... rest)
	{
		std::array<std::size_t, sizeof...(Rest) + 2> indices{
			first,
			second,
			static_cast<std::size_t>(rest)...
		};

		return buffer_[offset(shape_, strides_, indices)];
	}

	/** @brief Returns an element by multidimensional index with bounds checking. */
	template<typename... Rest>
	const T& operator()(std::size_t first, std::size_t second, Rest... rest) const
	{
		std::array<std::size_t, sizeof...(Rest) + 2> indices{
			first,
			second,
			static_cast<std::size_t>(rest)...
		};

		return buffer_[offset(shape_, strides_, indices)];
	}

	/** @brief Returns an element by multidimensional index from a vector with bounds checking. */
	T& operator()(const std::vector<std::size_t>& indices)
	{
		return buffer_[offset(shape_, strides_, indices)];
	}

	/** @brief Returns an element by multidimensional index from a vector with bounds checking. */
	const T& operator()(const std::vector<std::size_t>& indices) const
	{
		return buffer_[offset(shape_, strides_, indices)];
	}

	/** @brief Returns an element by rank-1 flat index with bounds checking. */
	T& at(std::ptrdiff_t index)
	{
		const std::size_t normalized =
			core::validation::normalize_index(index, size(), "Tensor flat index out of bounds.");
		return buffer_[normalized];
	}

	/** @brief Returns an element by rank-1 flat index with bounds checking. */
	const T& at(std::ptrdiff_t index) const
	{
		const std::size_t normalized =
			core::validation::normalize_index(index, size(), "Tensor flat index out of bounds.");
		return buffer_[normalized];
	}

	/** @brief Returns an element by multi-index with bounds checking. */
	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	T& at(std::ptrdiff_t first, std::ptrdiff_t second, Rest... rest)
	{
		std::array<std::ptrdiff_t, sizeof...(Rest) + 2> raw_indices{
			first,
			second,
			static_cast<std::ptrdiff_t>(rest)...
		};

		std::array<std::size_t, sizeof...(Rest) + 2> indices{};
		if (indices.size() != rank())
		{
			throw Exceptions::IndexError("Tensor multi-index rank must match tensor rank.");
		}

		for (std::size_t i = 0; i < indices.size(); ++i)
		{
			indices[i] = core::validation::normalize_index(
				raw_indices[i],
				shape()(i),
				"Tensor multi-index component is out of bounds.");
		}

		try
		{
			return buffer_[offset(shape_, strides_, indices)];
		}
		catch (const Exceptions::DimensionError&)
		{
			throw Exceptions::IndexError("Tensor multi-index rank must match tensor rank.");
		}
		catch (const Exceptions::IndexError&)
		{
			throw Exceptions::IndexError("Tensor multi-index component is out of bounds.");
		}
	}

	/** @brief Returns an element by multi-index with bounds checking. */
	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	const T& at(std::ptrdiff_t first, std::ptrdiff_t second, Rest... rest) const
	{
		std::array<std::ptrdiff_t, sizeof...(Rest) + 2> raw_indices{
			first,
			second,
			static_cast<std::ptrdiff_t>(rest)...
		};

		std::array<std::size_t, sizeof...(Rest) + 2> indices{};
		if (indices.size() != rank())
		{
			throw Exceptions::IndexError("Tensor multi-index rank must match tensor rank.");
		}

		for (std::size_t i = 0; i < indices.size(); ++i)
		{
			indices[i] = core::validation::normalize_index(
				raw_indices[i],
				shape()(i),
				"Tensor multi-index component is out of bounds.");
		}

		try
		{
			return buffer_[offset(shape_, strides_, indices)];
		}
		catch (const Exceptions::DimensionError&)
		{
			throw Exceptions::IndexError("Tensor multi-index rank must match tensor rank.");
		}
		catch (const Exceptions::IndexError&)
		{
			throw Exceptions::IndexError("Tensor multi-index component is out of bounds.");
		}
	}

	/** @brief Returns the first element. */
	T& front()
	{
		return buffer_.front();
	}

	/** @brief Returns the first element as a const reference. */
	const T& front() const
	{
		return buffer_.front();
	}

	/** @brief Returns the last element. */
	T& back()
	{
		return buffer_.back();
	}

	/** @brief Returns the last element as a const reference. */
	const T& back() const
	{
		return buffer_.back();
	}

	/** @brief Returns the raw data pointer. */
	[[nodiscard]] T* data() noexcept
	{
		return buffer_.data();
	}

	/** @brief Returns the raw data pointer as a const pointer. */
	[[nodiscard]] const T* data() const noexcept
	{
		return buffer_.data();
	}

	/** @brief Returns an iterator to the first element. */
	[[nodiscard]] iterator begin() noexcept
	{
		return buffer_.begin();
	}

	/** @brief Returns a const iterator to the first element. */
	[[nodiscard]] const_iterator begin() const noexcept
	{
		return buffer_.begin();
	}

	/** @brief Returns a const iterator to the first element. */
	[[nodiscard]] const_iterator cbegin() const noexcept
	{
		return buffer_.cbegin();
	}

	/** @brief Returns an iterator one past the last element. */
	[[nodiscard]] iterator end() noexcept
	{
		return buffer_.end();
	}

	/** @brief Returns a const iterator one past the last element. */
	[[nodiscard]] const_iterator end() const noexcept
	{
		return buffer_.end();
	}

	/** @brief Returns a const iterator one past the last element. */
	[[nodiscard]] const_iterator cend() const noexcept
	{
		return buffer_.cend();
	}

	/** @brief Returns a reverse iterator to the last element. */
	[[nodiscard]] reverse_iterator rbegin() noexcept
	{
		return buffer_.rbegin();
	}

	/** @brief Returns a const reverse iterator to the last element. */
	[[nodiscard]] const_reverse_iterator rbegin() const noexcept
	{
		return buffer_.rbegin();
	}

	/** @brief Returns a const reverse iterator to the last element. */
	[[nodiscard]] const_reverse_iterator crbegin() const noexcept
	{
		return buffer_.crbegin();
	}

	/** @brief Returns a reverse iterator before the first element. */
	[[nodiscard]] reverse_iterator rend() noexcept
	{
		return buffer_.rend();
	}

	/** @brief Returns a const reverse iterator before the first element. */
	[[nodiscard]] const_reverse_iterator rend() const noexcept
	{
		return buffer_.rend();
	}

	/** @brief Returns a const reverse iterator before the first element. */
	[[nodiscard]] const_reverse_iterator crend() const noexcept
	{
		return buffer_.crend();
	}

	/** @brief Fills every element with the same value. */
	void fill(const T& value)
	{
		buffer_.fill(value);
	}

	/** @brief Swaps the contents of two tensors. */
	void swap(Tensor& other) noexcept
	{
		shape_.swap(other.shape_);
		strides_.swap(other.strides_);
		buffer_.swap(other.buffer_);
	}
};

}
