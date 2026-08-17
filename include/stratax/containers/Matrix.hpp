#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::container {

template<typename T>
requires Numeric<T>
class Matrix : public core::ArrayBase<T>
{
private:
	static core::Shape initializer_shape(
		std::initializer_list<std::initializer_list<T>> list)
	{
		const std::size_t rows = list.size();
		const std::size_t cols = rows == 0 ? 0 : list.begin()->size();

		for (const auto& row : list)
		{
			if (row.size() != cols)
			{
				throw Exceptions::ShapeError(
					"Matrix initializer rows must all have the same number of columns.");
			}
		}

		return core::Shape{rows, cols};
	}

protected:
	using core::ArrayBase<T>::normalized_flat_offset;

public:
	using core::ArrayBase<T>::at;

	Matrix() : Matrix(0, 0) {}

	Matrix(std::size_t rows, std::size_t cols)
		: core::ArrayBase<T>(
			core::Shape({rows, cols}))
	{}

	Matrix(std::size_t rows, std::size_t cols, const T& value)
		: core::ArrayBase<T>(
			core::Shape({rows, cols}),
			value)
	{}

	explicit Matrix(const core::Shape& shape)
		: core::ArrayBase<T>(shape)
	{
		if (shape.rank() != 2)
		{
			throw Exceptions::ShapeError(
				"Matrix shape must be rank 2.");
		}
	}

	Matrix(std::initializer_list<std::initializer_list<T>> list)
		: core::ArrayBase<T>(initializer_shape(list))
	{
		std::size_t index = 0;

		for (const auto& row : list)
		{
			for (const auto& value : row)
			{
				(*this)[index++] = value;
			}
		}
	}

	[[nodiscard]] std::size_t rows() const noexcept
	{
		return this->shape()[0];
	}

	[[nodiscard]] std::size_t cols() const noexcept
	{
		return this->shape()[1];
	}

	T& operator()(std::size_t row, std::size_t col)
	{
    	return (*this)[row * cols() + col];
	}

	const T& operator()(std::size_t row, std::size_t col) const
	{
    	return (*this)[row * cols() + col];
	}

	T& at(std::ptrdiff_t row, std::ptrdiff_t col)
	{
		return (*this)[normalized_flat_offset(std::array<std::ptrdiff_t, 2>{row, col})];
	}

	const T& at(std::ptrdiff_t row, std::ptrdiff_t col) const
	{
		return (*this)[normalized_flat_offset(std::array<std::ptrdiff_t, 2>{row, col})];
	}

	void swap(Matrix& other) noexcept
	{
		core::ArrayBase<T>::swap(other);
	}
};

} // namespace stratax::container
