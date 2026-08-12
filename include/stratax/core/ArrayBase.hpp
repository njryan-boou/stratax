#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::core {

template<Numeric T>
class ArrayBase
{
private:

    void validate_empty() const
    {
        if (empty())
        {
            throw Exceptions::IndexError("Array is empty.");
        }
    }
public:
    ArrayBase() = default;
    ArrayBase(const ArrayBase&) = default;
    ArrayBase(ArrayBase&&) noexcept = default;
    ArrayBase& operator=(const ArrayBase&) = default;
    ArrayBase& operator=(ArrayBase&&) noexcept = default;
    ~ArrayBase() = default;

    using value_type = T;
    using iterator = typename Buffer<T>::iterator;
    using const_iterator = typename Buffer<T>::const_iterator;
    using reverse_iterator = typename Buffer<T>::reverse_iterator;
    using const_reverse_iterator = typename Buffer<T>::const_reverse_iterator;

    std::size_t size() const noexcept
    {
        return buffer_.size();
    }

    bool empty() const noexcept
    {
        return buffer_.empty();
    }

    std::size_t rank() const noexcept
	{
		return shape_.rank();
	}

    const Shape& shape() const noexcept
    {
        return shape_;
    }

    const Strides& strides() const noexcept
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

    T& front()
    {
        validate_empty();

        return buffer_.front();
    }

    const T& front() const
    {
        validate_empty();

        return buffer_.front();
    }

    T& back()
    {
        validate_empty();

        return buffer_.back();
    }

    const T& back() const
    {
        validate_empty();

        return buffer_.back();
    }

    T& operator[](std::size_t index) noexcept
    {
        return buffer_[index];
    }

    const T& operator[](std::size_t index) const noexcept
    {
        return buffer_[index];
    }

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

    void fill(const T& value)
    {
        buffer_.fill(value);
    }

    void swap(ArrayBase& other) noexcept
    {
        using std::swap;

        swap(shape_, other.shape_);
        swap(strides_, other.strides_);
        swap(buffer_, other.buffer_);
    }

protected:
    Buffer<T> buffer_;
    Shape shape_;
    Strides strides_;
};

}
