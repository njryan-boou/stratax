#pragma once

#include "Exceptions.hpp"

#include <cstddef>

namespace stratax::core {

class Slice
{
private:
    std::size_t start_;
    std::size_t stop_;

public:
    Slice(std::size_t start, std::size_t stop)
        : start_(start),
          stop_(stop)
    {
        if (start > stop) {
            throw Exceptions::IndexError("Slice start cannot be greater than stop");
        }
    }

    std::size_t start() const noexcept
    {
        return start_;
    }

    std::size_t stop() const noexcept
    {
        return stop_;
    }

    std::size_t size() const noexcept
    {
        return stop_ - start_;
    }

    bool empty() const noexcept
    {
        return size() == 0;
    }

    bool operator==(const Slice& other) const noexcept
    {
        return start_ == other.start_ && stop_ == other.stop_;
    }

    bool operator!=(const Slice& other) const noexcept
    {
        return !(*this == other);
    }
};

}
