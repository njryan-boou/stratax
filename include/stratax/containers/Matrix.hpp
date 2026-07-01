#pragma once

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Exceptions.hpp>

#include <initializer_list>
#include <cstddef>


namespace stratax::container {

template<typename T>
requires stratax::core::Numeric<T>
class Matrix
{
private:
    stratax::core::Shape shape_;
    stratax::core::Buffer<T> buffer_;

public:
    // ===== Type aliases =====

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using reference = T&;
    using const_reference = const T&;

    using pointer = T*;
    using const_pointer = const T*;

    using iterator = typename stratax::core::Buffer<T>::iterator;
    using const_iterator = typename stratax::core::Buffer<T>::const_iterator;

    using reverse_iterator = typename stratax::core::Buffer<T>::reverse_iterator;

    using const_reverse_iterator = typename stratax::core::Buffer<T>::const_reverse_iterator;

public:

    // ===== Constructors =====

    Matrix() noexcept = default;

    Matrix(size_type rows, size_type cols)
        : shape_{rows, cols},
          buffer_(rows * cols)
    {
    }

    Matrix(size_type rows, size_type cols, const T& value)
        : shape_{rows, cols},
          buffer_(rows * cols, value)
    {
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list)
    {
        size_type rows = list.size();
        size_type cols = (rows == 0) ? 0 : list.begin()->size();

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

        size_type index = 0;

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

    size_type size() const noexcept
    {
        return shape_.size();
    }

    bool empty() const noexcept
    {
        return buffer_.empty();
    }

    // ===== Shape =====

    size_type rows() const noexcept
    {
        return shape_[0];
    }

    size_type cols() const noexcept
    {
        return shape_[1];
    }

    const stratax::core::Shape& shape() const noexcept
    {
        return shape_;
    }

    // ===== Element Access =====

    reference operator()(size_type row, size_type col)
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

    const_reference operator()(size_type row, size_type col) const
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

    reference at(size_type row, size_type col)
    {
        return (*this)(row, col);
    }

    const_reference at(size_type row, size_type col) const
    {
        return (*this)(row, col);
    }

    reference front()
    {
        if (empty())
        {
            throw stratax::core::IndexError("Cannot access front of an empty matrix.");
        }

        return buffer_.front();
    }

    const_reference front() const
    {
        if (empty())
        {
            throw stratax::core::IndexError("Cannot access front of an empty matrix.");
        }

        return buffer_.front();
    }

    reference back()
    {
        if (empty())
        {
            throw stratax::core::IndexError("Cannot access back of an empty matrix.");
        }

        return buffer_.back();
    }

    const_reference back() const
    {
        if (empty())
        {
            throw stratax::core::IndexError("Cannot access back of an empty matrix.");
        }

        return buffer_.back();
    }

    // ===== Raw Data =====

    pointer data() noexcept
    {
        return buffer_.data();
    }

    const_pointer data() const noexcept
    {
        return buffer_.data();
    }

    // ===== Iterators =====

    iterator begin() noexcept
    {
        return buffer_.begin();
    }

    const_iterator begin() const noexcept
    {
        return buffer_.begin();
    }

    const_iterator cbegin() const noexcept
    {
        return buffer_.cbegin();
    }

    iterator end() noexcept
    {
        return buffer_.end();
    }

    const_iterator end() const noexcept
    {
        return buffer_.end();
    }

    const_iterator cend() const noexcept
    {
        return buffer_.cend();
    }

    reverse_iterator rbegin() noexcept
    {
        return buffer_.rbegin();
    }

    const_reverse_iterator rbegin() const noexcept
    {
        return buffer_.rbegin();
    }

    const_reverse_iterator crbegin() const noexcept
    {
        return buffer_.crbegin();
    }

    reverse_iterator rend() noexcept
    {
        return buffer_.rend();
    }

    const_reverse_iterator rend() const noexcept
    {
        return buffer_.rend();
    }

    const_reverse_iterator crend() const noexcept
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
