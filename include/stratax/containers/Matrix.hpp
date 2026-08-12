#pragma once

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <utility>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/validation/Validation.hpp>

namespace stratax::container {

/** @brief Stores a rank-2 Stratax array in row-major order. */
template<typename T = double>
requires Numeric<T>
class Matrix : public core::ArrayBase<T>
{
protected:
	using core::ArrayBase<T>::buffer_;
	using core::ArrayBase<T>::shape_;
	using core::ArrayBase<T>::strides_;

public:
	using core::ArrayBase<T>::operator();

	/** @brief Element type stored by the matrix. */
	using value_type = T;

	/** @brief Rebinds the matrix container to another element type. */
	template<typename U>
	using rebind = Matrix<U>;

	/** @brief Creates a default rank-2 empty matrix. */
	Matrix(): Matrix(0, 0) {}

	/** @brief Creates a rank-2 matrix with the given number of rows and columns. */
	Matrix(std::size_t rows, std::size_t cols)
	{
		shape_ = core::Shape({rows, cols}, core::Shape::allow_zero);
		strides_ = core::Strides(shape_);
		buffer_ = core::Buffer<T>(
			core::validation::checked_multiply(rows, cols, "Matrix size overflow"));
	}

	/** @brief Creates a matrix from a validated rank-2 shape. */
	explicit Matrix(const core::Shape& shape)
	{
		shape_ = core::validation::require_rank(shape, 2, "Shape must be rank 2");
		strides_ = core::Strides(shape_);
		buffer_ = core::Buffer<T>(core::validation::checked_multiply(
			shape_(0),
			shape_(1),
			"Matrix size overflow"));
	}

	/** @brief Creates a matrix and fills it with a value. */
	Matrix(std::size_t rows, std::size_t cols, const T& value)
	{
		shape_ = stratax::core::Shape({rows, cols}, stratax::core::Shape::allow_zero);
		strides_ = stratax::core::Strides(shape_);
		buffer_ = stratax::core::Buffer<T>(
			core::validation::checked_multiply(rows, cols, "Matrix size overflow"), value);
	}

	/** @brief Creates a matrix from a nested initializer list. */
	Matrix(std::initializer_list<std::initializer_list<T>> list)
	{
		std::size_t rows = list.size();
		std::size_t cols = (rows == 0) ? 0 : list.begin()->size();

		// Ensure all rows have the same length
		for (const auto& row : list)
		{
			if (row.size() != cols)
			{
				throw Exceptions::ShapeError("Matrix initializer rows must all have the same number of columns.");
			}
		}

		shape_ = stratax::core::Shape({rows, cols}, stratax::core::Shape::allow_zero);
		strides_ = stratax::core::Strides(shape_);
		buffer_ = stratax::core::Buffer<T>(
			core::validation::checked_multiply(rows, cols, "Matrix size overflow"));

		std::size_t index = 0;

		for (const auto& row : list)
		{
			for (const auto& value : row)
			{
				buffer_[index++] = value;
			}
		}
	}

	Matrix(const Matrix&) = default;
	Matrix(Matrix&&) noexcept = default;
	Matrix& operator=(const Matrix&) = default;
	Matrix& operator=(Matrix&&) noexcept = default;
	~Matrix() = default;

	/** @brief Returns the number of rows. */
	[[nodiscard]] std::size_t rows() const noexcept
	{
		return shape_(0);
	}

	/** @brief Returns the number of columns. */
	[[nodiscard]] std::size_t cols() const noexcept
	{
		return shape_(1);
	}

	/** @brief Returns an element by row and column with bounds checking. */
	T& operator()(std::size_t row, std::size_t col)
	{
		core::validation::require_index(row, rows(), "Row index out of bounds.");
		core::validation::require_index(col, cols(), "Column index out of bounds.");
		return buffer_[row * cols() + col];
	}

	/** @brief Returns an element by row and column with bounds checking. */
	const T& operator()(std::size_t row, std::size_t col) const
	{
		core::validation::require_index(row, rows(), "Row index out of bounds.");
		core::validation::require_index(col, cols(), "Column index out of bounds.");
		return buffer_[row * cols() + col];
	}

	/** @brief Returns an element by row and column. */
	T& at(std::ptrdiff_t row, std::ptrdiff_t col)
	{
		const std::size_t normalized_row =
			core::validation::normalize_index(row, rows(), "Row index out of bounds.");
		const std::size_t normalized_col =
			core::validation::normalize_index(col, cols(), "Column index out of bounds.");
		return (*this)(normalized_row, normalized_col);
	}

	/** @brief Returns an element by row and column. */
	const T& at(std::ptrdiff_t row, std::ptrdiff_t col) const
	{
		const std::size_t normalized_row =
			core::validation::normalize_index(row, rows(), "Row index out of bounds.");
		const std::size_t normalized_col =
			core::validation::normalize_index(col, cols(), "Column index out of bounds.");
		return (*this)(normalized_row, normalized_col);
	}

};

}
