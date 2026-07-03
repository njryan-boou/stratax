#pragma once

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Exceptions.hpp>

#include <initializer_list>
#include <cstddef>
#include <iterator>
#include <utility>

namespace stratax::container {

template<typename T>
requires stratax::core::Numeric<T>
class Matrix
{
private:
    stratax::core::Shape shape_;
    stratax::core::Buffer<T> buffer_;

public:

    // ===== Constructors =====

    Matrix() noexcept = default;

    Matrix(std::size_t rows, std::size_t cols)
        : shape_{rows, cols},
          buffer_(rows * cols)
    {
    }

    Matrix(std::size_t rows, std::size_t cols, const T& value)
        : shape_{rows, cols},
          buffer_(rows * cols, value)
    {
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list)
    {
        std::size_t rows = list.size();
        std::size_t cols = (rows == 0) ? 0 : list.begin()->size();

        // Ensure all rows have the same length
        for (const auto& row : list)
        {
            if (row.size() != cols)
            {
                throw stratax::core::StrataxError("All rows must have the same number of columns.");
            }
        }

        shape_ = stratax::core::Shape{rows, cols};
        buffer_ = stratax::core::Buffer<T>(rows * cols);

        std::size_t index = 0;

        for (const auto& row : list)
        {
            for (const auto& value : row)
            {
                buffer_[index++] = value;
            }
        }
    }

    // Rule of Five

    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;

    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    ~Matrix() = default;

    // ===== Capacity =====

    std::size_t size() const noexcept
    {
        return shape_.size();
    }

    bool empty() const noexcept
    {
        return buffer_.empty();
    }

    // ===== Shape =====

    std::size_t rows() const noexcept
    {
        return shape_[0];
    }

    std::size_t cols() const noexcept
    {
        return shape_[1];
    }

    const stratax::core::Shape& shape() const noexcept
    {
        return shape_;
    }

    // ===== Element Access =====

    T& operator()(std::size_t row, std::size_t col)
    {
        if (row >= rows())
        {
            throw stratax::core::IndexError("Row index out of bounds.");
        }

        if (col >= cols())
        {
            throw stratax::core::IndexError("Column index out of bounds.");
        }

        return buffer_[row * cols() + col];
    }

    const T& operator()(std::size_t row, std::size_t col) const
    {
        if (row >= rows())
        {
            throw stratax::core::IndexError("Row index out of bounds.");
        }

        if (col >= cols())
        {
            throw stratax::core::IndexError("Column index out of bounds.");
        }

        return buffer_[row * cols() + col];
    }

    T& at(std::size_t row, std::size_t col)
    {
        return (*this)(row, col);
    }

    const T& at(std::size_t row, std::size_t col) const
    {
        return (*this)(row, col);
    }

    T& front()
    {
        if (empty())
        {
            throw stratax::core::IndexError("Cannot access front of an empty matrix.");
        }

        return buffer_.front();
    }

    const T& front() const
    {
        if (empty())
        {
            throw stratax::core::IndexError("Cannot access front of an empty matrix.");
        }

        return buffer_.front();
    }

    T& back()
    {
        if (empty())
        {
            throw stratax::core::IndexError("Cannot access back of an empty matrix.");
        }

        return buffer_.back();
    }

    const T& back() const
    {
        if (empty())
        {
            throw stratax::core::IndexError("Cannot access back of an empty matrix.");
        }

        return buffer_.back();
    }

    // ===== Raw Data =====

    T* data() noexcept
    {
        return buffer_.data();
    }

    const T* data() const noexcept
    {
        return buffer_.data();
    }

    // ===== Iterators =====

    T* begin() noexcept
    {
        return buffer_.begin();
    }

    const T* begin() const noexcept
    {
        return buffer_.begin();
    }

    const T* cbegin() const noexcept
    {
        return buffer_.cbegin();
    }

    T* end() noexcept
    {
        return buffer_.end();
    }

    const T* end() const noexcept
    {
        return buffer_.end();
    }

    const T* cend() const noexcept
    {
        return buffer_.cend();
    }

    std::reverse_iterator<T*> rbegin() noexcept
    {
    return buffer_.rbegin();
    }

    std::reverse_iterator<const T*> rbegin() const noexcept
    {
        return buffer_.rbegin();
    }

    std::reverse_iterator<const T*> crbegin() const noexcept
    {
        return buffer_.crbegin();
    }

    std::reverse_iterator<T*> rend() noexcept
    {
        return buffer_.rend();
    }

    std::reverse_iterator<const T*> rend() const noexcept
    {
        return buffer_.rend();
    }

    std::reverse_iterator<const T*> crend() const noexcept
    {
        return buffer_.crend();
    }

    // ===== Utilities =====

    void fill(const T& value)
    {
        buffer_.fill(value);
    }

    void swap(Matrix& other) noexcept
    {
        using std::swap;

        swap(shape_, other.shape_);
        swap(buffer_, other.buffer_);
    }
};

}
