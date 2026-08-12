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
class Tensor : public core::ArrayBase<T>
{
protected:
	using core::ArrayBase<T>::allocate_from_shape;
	using core::ArrayBase<T>::buffer_;
	using core::ArrayBase<T>::flat_offset;
	using core::ArrayBase<T>::normalized_flat_offset;
	using core::ArrayBase<T>::set_shape_and_strides;
	using core::ArrayBase<T>::shape_;
	using core::ArrayBase<T>::strides_;

public:

	/** @brief Creates a tensor with default-initialized storage for a shape. */
	explicit Tensor(const core::Shape& shape)
    : core::ArrayBase<T>(shape)
{}

	/** @brief Creates a tensor from a shape and fills it with a value. */
	Tensor(const core::Shape& shape, const T& value)
    : core::ArrayBase<T>(shape, value)
{}

	Tensor() = default;
	Tensor(const Tensor&) = default;
	Tensor(Tensor&&) noexcept = default;
	Tensor& operator=(const Tensor&) = default;
	Tensor& operator=(Tensor&&) noexcept = default;
	~Tensor() = default;

	/** @brief Returns an element by multidimensional index with bounds checking. */
	template<typename... Rest>
	T& operator()(std::size_t first, std::size_t second, Rest... rest)
	{
		std::array<std::size_t, sizeof...(Rest) + 2> indices{
			first,
			second,
			static_cast<std::size_t>(rest)...
		};

		return buffer_[flat_offset(indices)];
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

		return buffer_[flat_offset(indices)];
	}

	/** @brief Returns an element by multidimensional index from a vector with bounds checking. */
	T& operator()(const std::vector<std::size_t>& indices)
	{
		return buffer_[flat_offset(indices)];
	}

	/** @brief Returns an element by multidimensional index from a vector with bounds checking. */
	const T& operator()(const std::vector<std::size_t>& indices) const
	{
		return buffer_[flat_offset(indices)];
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

		return buffer_[normalized_flat_offset(
			raw_indices,
			"Tensor multi-index rank must match tensor rank.",
			"Tensor multi-index component is out of bounds.")];
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

		return buffer_[normalized_flat_offset(
			raw_indices,
			"Tensor multi-index rank must match tensor rank.",
			"Tensor multi-index component is out of bounds.")];
	}

};

} // namespace stratax::container
