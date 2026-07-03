#pragma once

#include <cstddef>
#include <initializer_list>
#include <iterator>

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Exceptions.hpp>

namespace stratax::container {

template<typename T>
requires stratax::core::Numeric<T>
class Vector
{
private:
    stratax::core::Shape shape_;
    stratax::core::Buffer<T> buffer_;

public:
    // Constructors

    Vector() noexcept = default;

    explicit Vector(std::size_t size)
        : shape_{size},
          buffer_(size)
    {
    }

    Vector(std::size_t size, const T& value)
        : shape_{size},
          buffer_(size, value)
    {
    }

    Vector(std::initializer_list<T> list)
        : shape_{list.size()},
          buffer_(list)
    {
    }

    // Rule of Five

    Vector(const Vector&) = default;
    Vector(Vector&&) noexcept = default;

    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) noexcept = default;

    ~Vector() = default;


    // Element access

    T& operator[](std::size_t index) noexcept
    {
            return buffer_[index];
    }

    const T& operator[](std::size_t index) const noexcept
    {
            return buffer_[index];
    }

    T& at(std::size_t index)
    {
        if (index >= size())
        {
            throw stratax::core::IndexError("Vector index out of bounds");
        }

        return buffer_[index];
    }

    const T& at(std::size_t index) const
    {
        if (index >= size())
        {
            throw stratax::core::IndexError("Vector index out of bounds");
        }

        return buffer_[index];
    }

    T& front() noexcept
    {
        return buffer_.front();
    }

    const T& front() const noexcept
    {
        return buffer_.front();
    }

    T& back() noexcept
    {
        return buffer_.back();
    }

    const T& back() const noexcept
    {
        return buffer_.back();
    }

    // Size

    std::size_t size() const noexcept
    {
    return shape_.size();
    }

    bool empty() const noexcept
    {
    return buffer_.empty();
    }

    // Shape

    const stratax::core::Shape& shape() const noexcept
    {
    return shape_;
    }

    // Raw storage

    T* data() noexcept
    {
    return buffer_.data();
    }

    const T* data() const noexcept
    {
    return buffer_.data();
    }

    // Iterators

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

    // Utilities

    void fill(const T& value)
    {
        buffer_.fill(value);
    }

    void swap(Vector& other) noexcept
    {
        using std::swap;

        swap(shape_, other.shape_);
        swap(buffer_, other.buffer_);
    }

};

}