#pragma once

#include "Buffer.hpp"
#include "Shape.hpp"
#include "Exceptions.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <ostream>

namespace stratax::core {

class Strides
{
private:
    Buffer<std::size_t> buffer_;

public:
    Strides() noexcept = default;

    explicit Strides(const Shape& shape)
        : buffer_(shape.rank())
    {
        if (shape.empty()) {
            return;
        }

        buffer_[shape.rank() - 1] = 1;

        for (std::size_t i = shape.rank() - 1; i > 0; --i) {
            if (shape(i) != 0 &&
                buffer_[i] > std::numeric_limits<std::size_t>::max() / shape(i)) {
                throw Exceptions::DimensionError("Strides overflow for shape");
            }

            buffer_[i - 1] = buffer_[i] * shape(i);
        }
    }

    Strides(const Strides&) = default;
    Strides(Strides&&) noexcept = default;

    Strides& operator=(const Strides&) = default;
    Strides& operator=(Strides&&) noexcept = default;

    ~Strides() = default;

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
        return buffer_.size();
    }

    const std::size_t& operator()(std::size_t index) const
    {
        return buffer_[index];
    }

    const std::size_t& at(std::size_t index) const
    {
        if (index >= rank()) {
            throw Exceptions::IndexError("Strides index out of bounds");
        }

        return buffer_[index];
    }

    const std::size_t& front() const
    {
        return buffer_.front();
    }

    const std::size_t& back() const
    {
        return buffer_.back();
    }

    const std::size_t* data() const noexcept
    {
        return buffer_.data();
    }

    const std::size_t* begin() noexcept
    {
        return buffer_.begin();
    }

    const std::size_t* begin() const noexcept
    {
        return buffer_.begin();
    }

    const std::size_t* cbegin() const noexcept
    {
        return buffer_.cbegin();
    }

    const std::size_t* end() noexcept
    {
        return buffer_.end();
    }

    const std::size_t* end() const noexcept
    {
        return buffer_.end();
    }

    const std::size_t* cend() const noexcept
    {
        return buffer_.cend();
    }

    std::reverse_iterator<const std::size_t*> rbegin() noexcept
    {
        return buffer_.rbegin();
    }

    std::reverse_iterator<const std::size_t*> rbegin() const noexcept
    {
        return buffer_.rbegin();
    }

    std::reverse_iterator<const std::size_t*> crbegin() const noexcept
    {
        return buffer_.crbegin();
    }

    std::reverse_iterator<const std::size_t*> rend() noexcept
    {
        return buffer_.rend();
    }

    std::reverse_iterator<const std::size_t*> rend() const noexcept
    {
        return buffer_.rend();
    }

    std::reverse_iterator<const std::size_t*> crend() const noexcept
    {
        return buffer_.crend();
    }

    void swap(Strides& other) noexcept
    {
        buffer_.swap(other.buffer_);
    }

    bool operator==(const Strides& other) const noexcept
    {
        if (rank() != other.rank()) {
            return false;
        }

        for (std::size_t i = 0; i < rank(); ++i) {
            if (buffer_[i] != other.buffer_[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const Strides& other) const noexcept
    {
        return !(*this == other);
    }
};

inline std::ostream& operator<<(std::ostream& os, const Strides& stride)
{
    os << "(";

    bool first = true;
    for (std::size_t dim : stride)
    {
        if (!first)
            os << ", ";

        os << dim;
        first = false;
    }

    if (stride.rank() == 1)
    {
        os << ",";
    }

    os << ")";

    return os;
}

}
