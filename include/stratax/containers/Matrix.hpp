#pragma once

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Strides.hpp>

#include <initializer_list>
#include <cstddef>
#include <iterator>
#include <utility>

namespace stratax::container {

template<typename T>
requires Numeric<T>
class Matrix
{
private:
    core::Shape shape_;
    core::Strides strides_;
    core::Buffer<T> buffer_;

public:

    Matrix() noexcept = default;

    Matrix(const stratax::core::Shape& shape)
        : shape_(),
          buffer_()
    {
        shape_ = shape;
        buffer_ = stratax::core::Buffer<T>(shape.size());
    }

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
                throw Exceptions::StrataxError("All rows must have the same number of columns.");
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

    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;

    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    ~Matrix() = default;

    std::size_t size() const noexcept
    {
        return shape_.size();
    }

    bool empty() const noexcept
    {
        return buffer_.empty();
    }

    std::size_t rows() const noexcept
    {
        return shape_(0);
    }

    std::size_t cols() const noexcept
    {
        return shape_(1);
    }

    const stratax::core::Shape& shape() const noexcept
    {
        return shape_;
    }

    std::size_t rank() const noexcept
    {
        return shape_.rank();
    }

    T& operator()(std::size_t row, std::size_t col)
    {
        if (row >= rows())
        {
            throw Exceptions::IndexError("Row index out of bounds.");
        }

        if (col >= cols())
        {
            throw Exceptions::IndexError("Column index out of bounds.");
        }

        return buffer_[row * cols() + col];
    }

    const T& operator()(std::size_t row, std::size_t col) const
    {
        if (row >= rows())
        {
            throw Exceptions::IndexError("Row index out of bounds.");
        }

        if (col >= cols())
        {
            throw Exceptions::IndexError("Column index out of bounds.");
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
            throw Exceptions::IndexError("Cannot access front of an empty matrix.");
        }

        return buffer_.front();
    }

    const T& front() const
    {
        if (empty())
        {
            throw Exceptions::IndexError("Cannot access front of an empty matrix.");
        }

        return buffer_.front();
    }

    T& back()
    {
        if (empty())
        {
            throw Exceptions::IndexError("Cannot access back of an empty matrix.");
        }

        return buffer_.back();
    }

    const T& back() const
    {
        if (empty())
        {
            throw Exceptions::IndexError("Cannot access back of an empty matrix.");
        }

        return buffer_.back();
    }

    T* data() noexcept
    {
        return buffer_.data();
    }

    const T* data() const noexcept
    {
        return buffer_.data();
    }

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
