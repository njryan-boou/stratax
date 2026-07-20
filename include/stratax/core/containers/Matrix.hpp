#pragma once

#include <stratax/core/containers/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/containers/Shape.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/containers/Strides.hpp>
#include <stratax/core/validation/Validation.hpp>

#include <initializer_list>
#include <cstddef>
#include <iterator>
#include <utility>

namespace stratax::container {

template<typename T>
requires Numeric<T>
/**
 * @brief Stores a rank-2 Stratax array in row-major order.
 *
 * Matrix owns contiguous storage together with shape and stride
 * metadata, making it suitable for generic array operations and
 * two-dimensional indexing.
 *
 * @tparam T Numeric element type.
 */
/**
 * @ingroup containers
 */
class Matrix
{
private:
    core::Shape shape_;
    core::Strides strides_;
    core::Buffer<T> buffer_;

public:
    using value_type = T;

    template<typename U>
    using rebind = Matrix<U>;

    /**
     * @brief Creates a default rank-2 empty matrix.
     *
     * The default matrix has shape `(0, 0)` and owns no element storage.
     */
    Matrix(): Matrix(0, 0) {}

    /**
     * @brief Creates a rank-2 matrix with the given number of rows and columns.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    Matrix(std::size_t rows, std::size_t cols)
        : shape_({rows, cols}, core::Shape::allow_zero),
          strides_(shape_),
          buffer_(core::validation::checked_multiply(rows, cols, "Matrix size overflow"))
    {
    }

    /**
     * @brief Creates a matrix from a validated rank-2 shape.
     *
     * @param shape Source shape.
     *
     * @throws Exceptions::DimensionError If the shape is not rank 2.
     */
    explicit Matrix(const core::Shape& shape)
        : shape_(core::validation::require_rank(shape, 2, "Shape must be rank 2")),
          strides_(shape_),
          buffer_(core::validation::checked_multiply(
              shape_(0),
              shape_(1),
              "Matrix size overflow"))
    {
    }

    /**
     * @brief Creates a matrix and fills it with a value.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param value Value to copy into each element.
     */
    Matrix(std::size_t rows, std::size_t cols, const T& value)
        : shape_({rows, cols}, core::Shape::allow_zero),
          strides_(shape_),
          buffer_(core::validation::checked_multiply(rows, cols, "Matrix size overflow"), value)
    {
    }

    /**
     * @brief Creates a matrix from a nested initializer list.
     *
     * Rows are interpreted in row-major order and must all have the same length.
     *
     * @param list Row-major source values.
     *
     * @throws Exceptions::StrataxError If the rows are not rectangular.
     */
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

    /**
     * @brief Creates a copy of another matrix.
     */
    Matrix(const Matrix&) = default;

    /**
     * @brief Transfers ownership from another matrix.
     */
    Matrix(Matrix&&) noexcept = default;

    /**
     * @brief Replaces this matrix with a copy of another matrix.
     */
    Matrix& operator=(const Matrix&) = default;

    /**
     * @brief Replaces this matrix by taking ownership from another matrix.
     */
    Matrix& operator=(Matrix&&) noexcept = default;

    /**
     * @brief Destroys the matrix.
     */
    ~Matrix() = default;

    /**
     * @brief Returns the total number of elements in the matrix.
     *
     * @return Number of stored elements.
     */
    [[nodiscard]] std::size_t size() const noexcept
    {
        return shape_.elements();
    }

    /**
     * @brief Returns whether the matrix contains no elements.
     *
     * @return `true` when the matrix size is zero.
     */
    [[nodiscard]] bool empty() const noexcept
    {
        return buffer_.empty();
    }

    /**
     * @brief Returns the number of rows.
     *
     * @return Row count.
     */
    [[nodiscard]] std::size_t rows() const noexcept
    {
        return shape_(0);
    }

    /**
     * @brief Returns the number of columns.
     *
     * @return Column count.
     */
    [[nodiscard]] std::size_t cols() const noexcept
    {
        return shape_(1);
    }

    /**
     * @brief Returns the matrix shape.
     *
     * @return Shape describing the matrix.
     */
    const stratax::core::Shape& shape() const noexcept
    {
        return shape_;
    }

    /**
     * @brief Returns the matrix strides.
     *
     * @return Strides describing the matrix layout.
     */
    const stratax::core::Strides& strides() const noexcept
    {
        return strides_;
    }

    /**
     * @brief Returns the matrix rank.
     *
     * @return The number of dimensions, always 2 for valid matrices.
     */
    [[nodiscard]] std::size_t rank() const noexcept
    {
        return shape_.rank();
    }

    /**
     * @brief Returns an element by row and column with bounds checking.
     *
     * @param row Row index.
     * @param col Column index.
     *
     * @return Mutable reference to the indexed element.
     * @throws Exceptions::IndexError If either index is out of bounds.
     */
    T& operator()(std::size_t row, std::size_t col)
    {
        core::validation::require_index(row, rows(), "Row index out of bounds.");
        core::validation::require_index(col, cols(), "Column index out of bounds.");
        return buffer_[row * cols() + col];
    }

    /**
     * @brief Returns an element by row and column with bounds checking.
     *
     * @param row Row index.
     * @param col Column index.
     *
     * @return Const reference to the indexed element.
     * @throws Exceptions::IndexError If either index is out of bounds.
     */
    const T& operator()(std::size_t row, std::size_t col) const
    {
        core::validation::require_index(row, rows(), "Row index out of bounds.");
        core::validation::require_index(col, cols(), "Column index out of bounds.");
        return buffer_[row * cols() + col];
    }

    /**
     * @brief Returns a flat element without bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Mutable reference to the indexed element.
     * @warning The caller must ensure that the index is in range.
     */
    T& operator[](std::size_t index) noexcept
    {
        return buffer_[index];
    }

    /**
     * @brief Returns a flat element without bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Const reference to the indexed element.
     * @warning The caller must ensure that the index is in range.
     */
    const T& operator[](std::size_t index) const noexcept
    {
        return buffer_[index];
    }

    /**
     * @brief Returns an element by row and column.
     *
     * @param row Row index.
     * @param col Column index.
     *
     * @return Mutable reference to the indexed element.
     * @throws Exceptions::IndexError If either index is out of bounds.
     */
    T& at(std::ptrdiff_t row, std::ptrdiff_t col)
    {
        const std::size_t normalized_row =
            core::validation::normalize_index(row, rows(), "Row index out of bounds.");
        const std::size_t normalized_col =
            core::validation::normalize_index(col, cols(), "Column index out of bounds.");
        return (*this)(normalized_row, normalized_col);
    }

    /**
     * @brief Returns an element by row and column.
     *
     * @param row Row index.
     * @param col Column index.
     *
     * @return Const reference to the indexed element.
     * @throws Exceptions::IndexError If either index is out of bounds.
     */
    const T& at(std::ptrdiff_t row, std::ptrdiff_t col) const
    {
        const std::size_t normalized_row =
            core::validation::normalize_index(row, rows(), "Row index out of bounds.");
        const std::size_t normalized_col =
            core::validation::normalize_index(col, cols(), "Column index out of bounds.");
        return (*this)(normalized_row, normalized_col);
    }

    /**
     * @brief Returns the first element.
     *
     * @return Mutable reference to the first stored element.
     * @throws Exceptions::IndexError If the matrix is empty.
     */
    T& front()
    {
        if (empty())
        {
            throw Exceptions::IndexError("Matrix front cannot be accessed when the matrix is empty.");
        }

        return buffer_.front();
    }

    /**
     * @brief Returns the first element as a const reference.
     *
     * @return Const reference to the first stored element.
     * @throws Exceptions::IndexError If the matrix is empty.
     */
    const T& front() const
    {
        if (empty())
        {
            throw Exceptions::IndexError("Matrix front cannot be accessed when the matrix is empty.");
        }

        return buffer_.front();
    }

    /**
     * @brief Returns the last element.
     *
     * @return Mutable reference to the last stored element.
     * @throws Exceptions::IndexError If the matrix is empty.
     */
    T& back()
    {
        if (empty())
        {
            throw Exceptions::IndexError("Matrix back cannot be accessed when the matrix is empty.");
        }

        return buffer_.back();
    }

    /**
     * @brief Returns the last element as a const reference.
     *
     * @return Const reference to the last stored element.
     * @throws Exceptions::IndexError If the matrix is empty.
     */
    const T& back() const
    {
        if (empty())
        {
            throw Exceptions::IndexError("Matrix back cannot be accessed when the matrix is empty.");
        }

        return buffer_.back();
    }

    /**
     * @brief Returns the raw data pointer.
     *
     * @return Pointer to the first stored element, or `nullptr` when empty.
     */
    [[nodiscard]] T* data() noexcept
    {
        return buffer_.data();
    }

    /**
     * @brief Returns the raw data pointer as a const pointer.
     *
     * @return Pointer to the first stored element, or `nullptr` when empty.
     */
    [[nodiscard]] const T* data() const noexcept
    {
        return buffer_.data();
    }

    /**
     * @brief Returns an iterator to the first element.
     *
     * @return Iterator to the beginning of the matrix.
     */
    [[nodiscard]] T* begin() noexcept
    {
        return buffer_.begin();
    }

    /**
     * @brief Returns a const iterator to the first element.
     *
     * @return Const iterator to the beginning of the matrix.
     */
    [[nodiscard]] const T* begin() const noexcept
    {
        return buffer_.begin();
    }

    /**
     * @brief Returns a const iterator to the first element.
     *
     * @return Const iterator to the beginning of the matrix.
     */
    [[nodiscard]] const T* cbegin() const noexcept
    {
        return buffer_.cbegin();
    }

    /**
     * @brief Returns an iterator one past the last element.
     *
     * @return Iterator to the end of the matrix.
     */
    [[nodiscard]] T* end() noexcept
    {
        return buffer_.end();
    }

    /**
     * @brief Returns a const iterator one past the last element.
     *
     * @return Const iterator to the end of the matrix.
     */
    [[nodiscard]] const T* end() const noexcept
    {
        return buffer_.end();
    }

    /**
     * @brief Returns a const iterator one past the last element.
     *
     * @return Const iterator to the end of the matrix.
     */
    [[nodiscard]] const T* cend() const noexcept
    {
        return buffer_.cend();
    }

    /**
     * @brief Returns a reverse iterator to the last element.
        *
        * @return Reverse iterator starting at the last element.
     */
    [[nodiscard]] std::reverse_iterator<T*> rbegin() noexcept
    {
    return buffer_.rbegin();
    }

    /**
     * @brief Returns a const reverse iterator to the last element.
        *
        * @return Const reverse iterator starting at the last element.
     */
    [[nodiscard]] std::reverse_iterator<const T*> rbegin() const noexcept
    {
        return buffer_.rbegin();
    }

    /**
     * @brief Returns a const reverse iterator to the last element.
        *
        * @return Const reverse iterator starting at the last element.
     */
    [[nodiscard]] std::reverse_iterator<const T*> crbegin() const noexcept
    {
        return buffer_.crbegin();
    }

    /**
     * @brief Returns a reverse iterator before the first element.
        *
        * @return Reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<T*> rend() noexcept
    {
        return buffer_.rend();
    }

    /**
     * @brief Returns a const reverse iterator before the first element.
        *
        * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<const T*> rend() const noexcept
    {
        return buffer_.rend();
    }

    /**
     * @brief Returns a const reverse iterator before the first element.
        *
        * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<const T*> crend() const noexcept
    {
        return buffer_.crend();
    }

    /**
     * @brief Fills every element with the same value.
     *
     * @param value Value to assign to each element.
     */
    void fill(const T& value)
    {
        buffer_.fill(value);
    }

    /**
     * @brief Swaps the contents of two matrices.
     *
     * @param other Matrix to exchange state with.
     */
    void swap(Matrix& other) noexcept
    {
        using std::swap;

        swap(shape_, other.shape_);
        swap(strides_, other.strides_);
        swap(buffer_, other.buffer_);
    }
};

}

