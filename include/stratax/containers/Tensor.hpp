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
template<typename T = double>
requires Numeric<T>
class Tensor : public stratax::core::ArrayBase<T>
{
protected:
	using stratax::core::ArrayBase<T>::buffer_;
	using stratax::core::ArrayBase<T>::shape_;
	using stratax::core::ArrayBase<T>::strides_;

public:
	/** @brief Element type stored by the tensor. */
	using value_type = T;

	/** @brief Rebinds the tensor container to another element type. */
	template<typename U>
	using rebind = Tensor<U>;

	/** @brief Creates an empty tensor with rank 0. */
	Tensor() noexcept = default;

	/** @brief Creates a tensor with default-initialized storage for a shape. */
	explicit Tensor(const core::Shape& shape)
	{
		shape_ = shape;
		strides_ = core::Strides(shape_);
		buffer_ = core::Buffer<T>(shape_.elements());
	}

	/** @brief Creates a tensor from a shape and fills it with a value. */
	Tensor(const core::Shape& shape, const T& value)
	{
		shape_ = shape;
		strides_ = core::Strides(shape_);
		buffer_ = core::Buffer<T>(shape_.elements(), value);
	}

	Tensor(const Tensor&) = default;
	Tensor(Tensor&&) noexcept = default;
	Tensor& operator=(const Tensor&) = default;
	Tensor& operator=(Tensor&&) noexcept = default;
	~Tensor() = default;

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

	/** @brief Returns an element by multidimensional index with bounds checking. */
	template<typename... Rest>
	T& operator()(std::size_t first, std::size_t second, Rest... rest)
	{
		std::array<std::size_t, sizeof...(Rest) + 2> indices{
			first,
			second,
			static_cast<std::size_t>(rest)...
		};

		return buffer_[stratax::indexing::offset(shape_, strides_, indices)];
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

		return buffer_[stratax::indexing::offset(shape_, strides_, indices)];
	}

	/** @brief Returns an element by multidimensional index from a vector with bounds checking. */
	T& operator()(const std::vector<std::size_t>& indices)
	{
		return buffer_[stratax::indexing::offset(shape_, strides_, indices)];
	}

	/** @brief Returns an element by multidimensional index from a vector with bounds checking. */
	const T& operator()(const std::vector<std::size_t>& indices) const
	{
		return buffer_[stratax::indexing::offset(shape_, strides_, indices)];
	}

	/** @brief Returns an element by rank-1 flat index with bounds checking. */
	T& at(std::ptrdiff_t index)
	{
		const std::size_t normalized = stratax::indexing::normalize_index(index, this->size());
		return buffer_[normalized];
	}

	/** @brief Returns an element by rank-1 flat index with bounds checking. */
	const T& at(std::ptrdiff_t index) const
	{
		const std::size_t normalized = stratax::indexing::normalize_index(index, this->size());
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
		if (indices.size() != this->rank())
		{
			throw Exceptions::IndexError("Tensor multi-index rank must match tensor rank.");
		}

		for (std::size_t i = 0; i < indices.size(); ++i)
		{
			indices[i] = stratax::indexing::normalize_index(
				raw_indices[i],
				this->shape()[i]);
		}

		try
		{
			return buffer_[stratax::indexing::offset(this->shape_, this->strides_, indices)];
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
		if (indices.size() != this->rank())
		{
			throw Exceptions::IndexError("Tensor multi-index rank must match tensor rank.");
		}

		for (std::size_t i = 0; i < indices.size(); ++i)
		{
			indices[i] = stratax::indexing::normalize_index(
				raw_indices[i],
				this->shape()[i]);
		}

		try
		{
			return buffer_[stratax::indexing::offset(this->shape_, this->strides_, indices)];
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

};

}
