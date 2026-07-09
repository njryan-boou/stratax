#pragma once

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Validation.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/ops/Indexing.hpp>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace stratax::container {

template<typename T>
requires Numeric<T>
class Tensor
{
private:
    core::Shape shape_;
    core::Strides strides_;
    core::Buffer<T> buffer_;

public:
    using value_type = T;

    template<typename U>
    using rebind = Tensor<U>;

    Tensor() noexcept = default;

    explicit Tensor(const core::Shape& shape)
        : shape_(shape),
          strides_(shape_),
          buffer_(shape_.elements())
    {
    }

    Tensor(const core::Shape& shape, const T& value)
        : shape_(shape),
          strides_(shape_),
          buffer_(shape_.elements(), value)
    {
    }

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;

    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    ~Tensor() = default;

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

    const core::Shape& shape() const noexcept
    {
        return shape_;
    }

    const core::Strides& strides() const noexcept
    {
        return strides_;
    }

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

    template<typename... Rest>
    T& operator()(std::size_t first, std::size_t second, Rest... rest)
    {
        std::array<std::size_t, sizeof...(Rest) + 2> indices{
            first,
            second,
            static_cast<std::size_t>(rest)...
        };

        return buffer_[offset(shape_, strides_, indices)];
    }

    template<typename... Rest>
    const T& operator()(std::size_t first, std::size_t second, Rest... rest) const
    {
        std::array<std::size_t, sizeof...(Rest) + 2> indices{
            first,
            second,
            static_cast<std::size_t>(rest)...
        };

        return buffer_[offset(shape_, strides_, indices)];
    }

    T& at(std::size_t index)
    {
        if (index >= size())
        {
            throw Exceptions::IndexError("Tensor flat index out of bounds.");
        }

        return buffer_[index];
    }

    const T& at(std::size_t index) const
    {
        if (index >= size())
        {
            throw Exceptions::IndexError("Tensor flat index out of bounds.");
        }

        return buffer_[index];
    }

    template<typename... Rest>
    T& at(std::size_t first, std::size_t second, Rest... rest)
    {
        std::array<std::size_t, sizeof...(Rest) + 2> indices{
            first,
            second,
            static_cast<std::size_t>(rest)...
        };

        try
        {
            return buffer_[offset(shape_, strides_, indices)];
        }
        catch (const Exceptions::DimensionError&)
        {
            throw Exceptions::IndexError("Tensor multi-index rank mismatch.");
        }
        catch (const Exceptions::IndexError&)
        {
            throw Exceptions::IndexError("Tensor multi-index out of bounds.");
        }
    }

    template<typename... Rest>
    const T& at(std::size_t first, std::size_t second, Rest... rest) const
    {
        std::array<std::size_t, sizeof...(Rest) + 2> indices{
            first,
            second,
            static_cast<std::size_t>(rest)...
        };

        try
        {
            return buffer_[offset(shape_, strides_, indices)];
        }
        catch (const Exceptions::DimensionError&)
        {
            throw Exceptions::IndexError("Tensor multi-index rank mismatch.");
        }
        catch (const Exceptions::IndexError&)
        {
            throw Exceptions::IndexError("Tensor multi-index out of bounds.");
        }
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

    T* data() noexcept
    {
        return buffer_.data();
    }

    const T* data() const noexcept
    {
        return buffer_.data();
    }

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
