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

		return core::Shape({rows, cols});
	}

protected:
	using core::ArrayBase<T>::buffer_;
	using core::ArrayBase<T>::normalized_flat_offset;
	using core::ArrayBase<T>::shape_;
	using core::ArrayBase<T>::flat_offset;

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
		: core::ArrayBase<T>(
			core::validation::require_rank(
				shape,
				2,
				"Matrix requires a rank-2 shape."))
	{}

	Matrix(std::initializer_list<std::initializer_list<T>> list)
		: core::ArrayBase<T>(initializer_shape(list))
	{
		std::size_t index = 0;

		for (const auto& row : list)
		{
			for (const auto& value : row)
			{
				buffer_[index++] = value;
			}
		}
	}

	[[nodiscard]] std::size_t rows() const noexcept
	{
		return shape_[0];
	}

	[[nodiscard]] std::size_t cols() const noexcept
	{
		return shape_[1];
	}

	T& operator()(std::size_t row, std::size_t col)
	{
		return buffer_[row * cols() + col];
	}

	const T& operator()(std::size_t row, std::size_t col) const
	{
		return buffer_[row * cols() + col];
	}

	T& at(std::ptrdiff_t row, std::ptrdiff_t col)
	{
		return buffer_[normalized_flat_offset(std::array<std::ptrdiff_t, 2>{row, col})];
	}

	const T& at(std::ptrdiff_t row, std::ptrdiff_t col) const
	{
		return buffer_[normalized_flat_offset(std::array<std::ptrdiff_t, 2>{row, col})];
	}

	void swap(Matrix& other) noexcept
	{
		using std::swap;

		swap(this->shape_, other.shape_);
		swap(this->strides_, other.strides_);
		swap(this->buffer_, other.buffer_);
	}

	friend void swap(Matrix& lhs, Matrix& rhs) noexcept
	{
		lhs.swap(rhs);
	}
};

} // namespace stratax::container
