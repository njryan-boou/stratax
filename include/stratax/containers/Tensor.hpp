#pragma once

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Strides.hpp>

#include <cstddef>
#include <utility>

namespace stratax::container {

template<typename T>
requires core::Numeric<T>
class Tensor
{
private:
    core::Shape shape_;
    core::Strides strides_;
    core::Buffer<T> buffer_;

public:
    // Constructors

    Tensor() noexcept = default;

    explicit Tensor(const core::Shape& shape)
        : shape_(shape),
          strides_(shape),
          buffer_(shape.size())
    {
    }

    Tensor(const core::Shape& shape, const T& value)
        : shape_(shape),
          strides_(shape),
          buffer_(shape.size(), value)
    {
    }

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;

    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    ~Tensor() = default;

    // Capacity

    std::size_t size() const noexcept
    {
        return buffer_.size();
    }

    bool empty() const noexcept
    {
        return buffer_.empty();
    }

    // Shape

    std::size_t rank() const noexcept
    {
        return shape_.rank();
    }

    const core::Shape& shape() const noexcept
    {
        return shape_;
    }

    const core::Strides& strides() const noexcept
    {
        return strides_;
    }

    // Element access (flat)

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
        if (index >= size()) {
            throw core::IndexError("Tensor index out of bounds");
        }

        return buffer_[index];
    }

    const T& at(std::size_t index) const
    {
        if (index >= size()) {
            throw core::IndexError("Tensor index out of bounds");
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

    // Raw data

    T* data() noexcept
    {
        return buffer_.data();
    }

    const T* data() const noexcept
    {
        return buffer_.data();
    }

    // Iterators

    auto begin() noexcept
    {
        return buffer_.begin();
    }

    auto begin() const noexcept
    {
        return buffer_.begin();
    }

    auto cbegin() const noexcept
    {
        return buffer_.cbegin();
    }

    auto end() noexcept
    {
        return buffer_.end();
    }

    auto end() const noexcept
    {
        return buffer_.end();
    }

    auto cend() const noexcept
    {
        return buffer_.cend();
    }

    auto rbegin() noexcept
    {
        return buffer_.rbegin();
    }

    auto rbegin() const noexcept
    {
        return buffer_.rbegin();
    }

    auto crbegin() const noexcept
    {
        return buffer_.crbegin();
    }

    auto rend() noexcept
    {
        return buffer_.rend();
    }

    auto rend() const noexcept
    {
        return buffer_.rend();
    }

    auto crend() const noexcept
    {
        return buffer_.crend();
    }

    // Utilities

    void fill(const T& value)
    {
        buffer_.fill(value);
    }

    void swap(Tensor& other) noexcept
    {
        shape_.swap(other.shape_);
        strides_.swap(other.strides_);
        buffer_.swap(other.buffer_);
    }
};

}
