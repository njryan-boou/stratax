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
template<typename T>
requires Numeric<T>
class Matrix : public core::ArrayBase<T>
{
public:
	/** @brief Element type stored by the matrix. */
	using value_type = T;

	/** @brief Mutable iterator over matrix elements. */
	using iterator = typename core::Buffer<T>::iterator;

	/** @brief Const iterator over matrix elements. */
	using const_iterator = typename core::Buffer<T>::const_iterator;

	/** @brief Mutable reverse iterator over matrix elements. */
	using reverse_iterator = typename core::Buffer<T>::reverse_iterator;

	/** @brief Const reverse iterator over matrix elements. */
	using const_reverse_iterator = typename core::Buffer<T>::const_reverse_iterator;

	/** @brief Rebinds the matrix container to another element type. */
	template<typename U>
	using rebind = Matrix<U>;

	/** @brief Creates a default rank-2 empty matrix. */
	Matrix(): Matrix(0, 0) {}

	/** @brief Creates a rank-2 matrix with the given number of rows and columns. */
	Matrix(std::size_t rows, std::size_t cols)
		: shape_({rows, cols}, core::Shape::allow_zero),
		  strides_(shape_),
		  buffer_(core::validation::checked_multiply(rows, cols, "Matrix size overflow"))
	{
	}

	/** @brief Creates a matrix from a validated rank-2 shape. */
	explicit Matrix(const core::Shape& shape)
		: shape_(core::validation::require_rank(shape, 2, "Shape must be rank 2")),
		  strides_(shape_),
		  buffer_(core::validation::checked_multiply(
			  shape_(0),
			  shape_(1),
			  "Matrix size overflow"))
	{
	}

	/** @brief Creates a matrix and fills it with a value. */
	Matrix(std::size_t rows, std::size_t cols, const T& value)
		: shape_({rows, cols}, core::Shape::allow_zero),
		  strides_(shape_),
		  buffer_(core::validation::checked_multiply(rows, cols, "Matrix size overflow"), value)
	{
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

	/** @brief Creates a copy of another matrix. */
	Matrix(const Matrix&) = default;

	/** @brief Transfers ownership from another matrix. */
	Matrix(Matrix&&) noexcept = default;

	/** @brief Replaces this matrix with a copy of another matrix. */
	Matrix& operator=(const Matrix&) = default;

	/** @brief Replaces this matrix by taking ownership from another matrix. */
	Matrix& operator=(Matrix&&) noexcept = default;

	/** @brief Destroys the matrix. */
	~Matrix() = default;

	/** @brief Returns the total number of elements in the matrix. */
	[[nodiscard]] std::size_t size() const noexcept
	{
		return shape_.elements();
	}

	/** @brief Returns whether the matrix contains no elements. */
	[[nodiscard]] bool empty() const noexcept
	{
		return buffer_.empty();
	}

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

	/** @brief Returns the matrix shape. */
	const stratax::core::Shape& shape() const noexcept
	{
		return shape_;
	}

	/** @brief Returns the matrix strides. */
	const stratax::core::Strides& strides() const noexcept
	{
		return strides_;
	}

	/** @brief Returns the matrix rank. */
	[[nodiscard]] std::size_t rank() const noexcept
	{
		return shape_.rank();
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

	/** @brief Returns the first element. */
	T& front()
	{
		if (empty())
		{
			throw Exceptions::IndexError("Matrix front cannot be accessed when the matrix is empty.");
		}

		return buffer_.front();
	}

	/** @brief Returns the first element as a const reference. */
	const T& front() const
	{
		if (empty())
		{
			throw Exceptions::IndexError("Matrix front cannot be accessed when the matrix is empty.");
		}

		return buffer_.front();
	}

	/** @brief Returns the last element. */
	T& back()
	{
		if (empty())
		{
			throw Exceptions::IndexError("Matrix back cannot be accessed when the matrix is empty.");
		}

		return buffer_.back();
	}

	/** @brief Returns the last element as a const reference. */
	const T& back() const
	{
		if (empty())
		{
			throw Exceptions::IndexError("Matrix back cannot be accessed when the matrix is empty.");
		}

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

	/** @brief Swaps the contents of two matrices. */
	void swap(Matrix& other) noexcept
	{
		using std::swap;

		swap(shape_, other.shape_);
		swap(strides_, other.strides_);
		swap(buffer_, other.buffer_);
	}
};

}
