#pragma once

#include <cstddef>
#include <initializer_list>
#include <iterator>

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/Exceptions.hpp>

namespace stratax::container {

template<typename T>
requires Numeric<T>
class Vector
{
private:
    core::Shape shape_;
    core::Strides strides_;
    core::Buffer<T> buffer_;

    static const core::Shape& validate_shape(const core::Shape& shape)
    {
        if (shape.rank() != 1)
        {
            throw Exceptions::DimensionError("Shape must be rank 1");
        }

        return shape;
    }

public:
    using value_type = T;

    template<typename U>
    using rebind = Vector<U>;

    Vector() noexcept = default;

    explicit Vector(std::size_t size) : shape_{size}, strides_(shape_), buffer_(size)
    {
    }

    explicit Vector(const core::Shape& shape)
        : shape_(validate_shape(shape)), strides_(shape_), buffer_(shape_.elements())
    {
    }

    Vector(std::size_t size, const T& value) : shape_{size}, strides_(shape_), buffer_(size, value)
    {
    }

    Vector(std::initializer_list<T> list) : shape_{list.size()}, strides_(shape_), buffer_(list)
    {
    }

    // Rule of Five

    Vector(const Vector&) = default;
    Vector(Vector&&) noexcept = default;

    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) noexcept = default;

    ~Vector() = default;

    T& operator()(std::size_t index) noexcept
    {
            return buffer_[index];
    }

    const T& operator()(std::size_t index) const noexcept
    {
            return buffer_[index];
    }

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
            throw Exceptions::IndexError("Vector index out of bounds.");
        }

        return buffer_[index];
    }

    const T& at(std::size_t index) const
    {
        if (index >= size())
        {
            throw Exceptions::IndexError("Vector index out of bounds.");
        }

        return buffer_[index];
    }

    T& front()
    {
        return buffer_.front();
    }

    const T& front() const
    {
        return buffer_.front();
    }

    T& back()
    {
        return buffer_.back();
    }

    const T& back() const
    {
        return buffer_.back();
    }

    std::size_t size() const noexcept
    {
    return shape_.elements();
    }

    std::size_t rank() const noexcept
    {
        return shape_.rank();
    }

    bool empty() const noexcept
    {
    return buffer_.empty();
    }

    const stratax::core::Shape& shape() const noexcept
    {
    return shape_;
    }

    const stratax::core::Strides& strides() const noexcept
    {
        return strides_;
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

    void swap(Vector& other) noexcept
    {
        using std::swap;

        swap(shape_, other.shape_);
        swap(strides_, other.strides_);
        swap(buffer_, other.buffer_);
    }

};

}
