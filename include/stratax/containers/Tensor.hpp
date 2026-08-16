// TODO: make normalize flat offset more explicit

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>
#include <string>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Shape.hpp>

namespace stratax::container {

template<typename T>
requires Numeric<T>
class Tensor : public core::ArrayBase<T>
{
protected:
	using core::ArrayBase<T>::buffer_;
	using core::ArrayBase<T>::flat_offset;
	using core::ArrayBase<T>::normalized_flat_offset;

public:
	using core::ArrayBase<T>::at;

	Tensor() : Tensor(core::Shape({0})) {}

	explicit Tensor(const core::Shape& shape)
		: core::ArrayBase<T>(shape)
	{}

	Tensor(const core::Shape& shape, const T& value)
		: core::ArrayBase<T>(shape, value)
	{}

	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	T& operator()(std::size_t first, std::size_t second, Rest... rest)
	{
		std::array<std::size_t, sizeof...(Rest) + 2> indices{
			first,
			second,
			static_cast<std::size_t>(rest)...
		};

		return buffer_[flat_offset(indices)];
	}

	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	const T& operator()(std::size_t first, std::size_t second, Rest... rest) const
	{
		std::array<std::size_t, sizeof...(Rest) + 2> indices{
			first,
			second,
			static_cast<std::size_t>(rest)...
		};

		return buffer_[flat_offset(indices)];
	}

	T& operator()(const std::vector<std::size_t>& indices)
	{
		return buffer_[flat_offset(indices)];
	}

	const T& operator()(const std::vector<std::size_t>& indices) const
	{
		return buffer_[flat_offset(indices)];
	}

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

	T& at(const std::vector<std::ptrdiff_t>& raw_indices)
	{
		return buffer_[normalized_flat_offset(
			raw_indices,
			"Tensor multi-index rank must match tensor rank.",
			"Tensor multi-index component is out of bounds.")];
	}

	const T& at(const std::vector<std::ptrdiff_t>& raw_indices) const
	{
		return buffer_[normalized_flat_offset(
			raw_indices,
			"Tensor multi-index rank must match tensor rank.",
			"Tensor multi-index component is out of bounds.")];
	}

	void swap(Tensor& other) noexcept
	{
		using std::swap;

		swap(this->shape_, other.shape_);
		swap(this->strides_, other.strides_);
		swap(this->buffer_, other.buffer_);
	}

	friend void swap(Tensor& lhs, Tensor& rhs) noexcept
	{
		lhs.swap(rhs);
	}
};

} // namespace stratax::container
