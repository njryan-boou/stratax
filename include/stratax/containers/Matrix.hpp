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

/**
 * @brief Two-dimensional owning array of numeric values.
 *
 * Matrix specializes core::ArrayBase for rank-two data arranged in row-major
 * order. It adds row and column metadata queries plus two-dimensional checked
 * and unchecked access while retaining the base class's flat container API.
 *
 * @tparam T Numeric element type satisfying the Numeric concept.
 *
 * @par Invariants
 * - `rank() == 2`, including when either dimension is zero.
 * - `shape() == Shape{rows(), cols()}`.
 * - `size() == rows() * cols()` when that product is representable.
 * - Elements are contiguous in row-major order.
 *
 * @par Example
 * @code
 * stratax::container::Matrix<double> matrix{
 *     {1.0, 2.0, 3.0},
 *     {4.0, 5.0, 6.0}
 * };
 *
 * matrix(1, 2);   // 6.0; unchecked access
 * matrix.at(-1, 0); // 4.0; checked access with negative indices
 * @endcode
 *
 * @see core::ArrayBase
 * @see core::Shape
 */
template<typename T>
requires Numeric<T>
class Matrix : public core::ArrayBase<T>
{
public:
	/** @brief Stored element type inherited from ArrayBase. */
	using value_type = typename core::ArrayBase<T>::value_type;
	/** @brief Unsigned type used for element counts and indices. */
	using size_type = typename core::ArrayBase<T>::size_type;
	/** @brief Signed type used for checked indices and iterator distances. */
	using difference_type = typename core::ArrayBase<T>::difference_type;
	/** @brief Mutable element reference type. */
	using reference = typename core::ArrayBase<T>::reference;
	/** @brief Read-only element reference type. */
	using const_reference = typename core::ArrayBase<T>::const_reference;
	/** @brief Mutable element pointer type. */
	using pointer = typename core::ArrayBase<T>::pointer;
	/** @brief Read-only element pointer type. */
	using const_pointer = typename core::ArrayBase<T>::const_pointer;
	/** @brief Mutable contiguous random-access iterator type. */
	using iterator = typename core::ArrayBase<T>::iterator;
	/** @brief Read-only contiguous random-access iterator type. */
	using const_iterator = typename core::ArrayBase<T>::const_iterator;
	/** @brief Mutable reverse iterator type. */
	using reverse_iterator = typename core::ArrayBase<T>::reverse_iterator;
	/** @brief Read-only reverse iterator type. */
	using const_reverse_iterator = typename core::ArrayBase<T>::const_reverse_iterator;

private:
	/**
	 * @brief Validates a nested initializer and derives its rank-two shape.
	 * @param list Rows to inspect.
	 * @return Shape containing the number of rows and columns.
	 * @throws Exceptions::ShapeError If the rows have different lengths.
	 * @complexity O(list.size()).
	 */
	static core::Shape initializer_shape(
		std::initializer_list<std::initializer_list<value_type>> list)
	{
		const size_type rows = list.size();
		const size_type cols = rows == 0 ? 0 : list.begin()->size();

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

	/**
	 * @brief Verifies that a shape can describe a matrix.
	 * @param shape Candidate shape.
	 * @return @p shape unchanged when it has rank two.
	 * @throws Exceptions::ShapeError If `shape.rank() != 2`.
	 * @complexity O(1).
	 */
	static const core::Shape& validate_shape(const core::Shape& shape)
    {
        if (shape.rank() != 2) {
            throw Exceptions::ShapeError(
                "Matrix shape must be rank 2.");
        }

        return shape;
    }

protected:
	/** @brief Exposes ArrayBase's checked multidimensional offset helper. */
	using core::ArrayBase<T>::normalized_flat_offset;

public:
	/** @brief Retains the inherited checked flat `at(index)` overloads. */
	using core::ArrayBase<T>::at;

	/**
	 * @brief Constructs an empty matrix with shape `{0, 0}`.
	 * @complexity O(1).
	 */
	Matrix() : Matrix(0, 0) {}

	/**
	 * @brief Constructs a value-initialized matrix with the requested dimensions.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @throws Exceptions::DimensionError If the element or stride count overflows.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown while value-initializing a value_type.
	 * @complexity O(rows * cols).
	 */
	Matrix(size_type rows, size_type cols)
		: core::ArrayBase<T>(core::Shape{rows, cols})
	{}

	/**
	 * @brief Constructs a matrix filled with copies of @p value.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @param value Value copied into every element.
	 * @throws Exceptions::DimensionError If the element or stride count overflows.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown by value_type's copy constructor.
	 * @complexity O(rows * cols).
	 */
	Matrix(size_type rows, size_type cols, const_reference value)
		: core::ArrayBase<T>(core::Shape{rows, cols}, value)
	{}

	/**
	 * @brief Constructs a value-initialized matrix from a rank-two shape.
	 * @param shape Shape whose dimensions specify rows and columns.
	 * @throws Exceptions::ShapeError If `shape.rank() != 2`.
	 * @throws Exceptions::DimensionError If the element or stride count overflows.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown while value-initializing a value_type.
	 * @complexity O(shape.elements()).
	 */
	explicit Matrix(const core::Shape& shape)
		: core::ArrayBase<T>(validate_shape(shape))
	{}

	/**
	 * @brief Constructs a matrix by copying a rectangular nested initializer.
	 *
	 * Rows retain their input order and each row occupies one contiguous segment
	 * of the resulting row-major storage. An empty outer list creates shape
	 * `{0, 0}`; non-empty rows may consistently contain zero columns.
	 *
	 * @param list Rectangular sequence of matrix rows.
	 * @throws Exceptions::ShapeError If the rows have different lengths.
	 * @throws Exceptions::DimensionError If the element or stride count overflows.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown while constructing or assigning value_type.
	 * @complexity O(rows * cols).
	 */
	Matrix(std::initializer_list<std::initializer_list<value_type>> list)
		: core::ArrayBase<T>(initializer_shape(list))
	{
		size_type index = 0;

		for (const auto& row : list)
		{
			for (const auto& value : row)
			{
				(*this)[index++] = value;
			}
		}
	}

	/** @brief Returns the number of rows. @complexity O(1). */
	[[nodiscard]] size_type rows() const noexcept {return this->shape()[0];}
	/** @brief Returns the number of columns. @complexity O(1). */
	[[nodiscard]] size_type cols() const noexcept {return this->shape()[1];}

	/**
	 * @brief Returns an element using unchecked row and column indices.
	 * @param row Zero-based row index.
	 * @param col Zero-based column index.
	 * @pre `row < rows()` and `col < cols()`.
	 * @complexity O(1).
	 */
	reference operator()(size_type row, size_type col) {return (*this)[row * cols() + col];}
	/**
	 * @brief Returns an element using unchecked row and column indices.
	 * @param row Zero-based row index.
	 * @param col Zero-based column index.
	 * @pre `row < rows()` and `col < cols()`.
	 * @complexity O(1).
	 */
	const_reference operator()(size_type row, size_type col) const {return (*this)[row * cols() + col];}

	/**
	 * @brief Returns an element using checked, Python-style indices.
	 * @param row Row index in `[-rows(), rows())`.
	 * @param col Column index in `[-cols(), cols())`.
	 * @throws Exceptions::IndexError If either index is out of bounds.
	 * @complexity O(1).
	 */
	reference at(difference_type row, difference_type col) {return (*this)[normalized_flat_offset(std::array<difference_type, 2>{row, col})];}
	/**
	 * @brief Returns an element using checked, Python-style indices.
	 * @param row Row index in `[-rows(), rows())`.
	 * @param col Column index in `[-cols(), cols())`.
	 * @throws Exceptions::IndexError If either index is out of bounds.
	 * @complexity O(1).
	 */
	const_reference at(difference_type row, difference_type col) const {return (*this)[normalized_flat_offset(std::array<difference_type, 2>{row, col})];}

	/**
	 * @brief Exchanges storage and layout metadata with @p other.
	 * @param other Matrix to exchange with this matrix.
	 * @complexity O(1).
	 */
	void swap(Matrix& other) noexcept {core::ArrayBase<T>::swap(other);}
	/**
	 * @brief Exchanges two matrices using argument-dependent lookup.
	 * @param lhs First matrix.
	 * @param rhs Second matrix.
	 * @complexity O(1).
	 */
	friend void swap(Matrix& lhs, Matrix& rhs) noexcept {lhs.swap(rhs);}
};

} // namespace stratax::container
