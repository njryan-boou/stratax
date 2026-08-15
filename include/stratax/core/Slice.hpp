#pragma once

#include <cstddef>

#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::core {

class Slice
{
private:
    std::ptrdiff_t start_;
    std::ptrdiff_t stop_;
    std::ptrdiff_t step_;

public:
    Slice(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step = 1)
        : start_(start),
          stop_(stop),
          step_(step)
    {
        if (step == 0) {
            throw Exceptions::IndexError("Slice step cannot be zero.");
        }
    }

    [[nodiscard]] std::ptrdiff_t start() const noexcept
    {
        return start_;
    }

    [[nodiscard]] std::ptrdiff_t stop() const noexcept
    {
        return stop_;
    }

    [[nodiscard]] std::ptrdiff_t step() const noexcept
    {
        return step_;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        if (step_ > 0)
        {
            if (start_ >= stop_)
            {
                return 0;
            }

            const std::ptrdiff_t distance = stop_ - start_;
            return static_cast<std::size_t>((distance + step_ - 1) / step_);
        }

        if (start_ <= stop_)
        {
            return 0;
        }

        const std::ptrdiff_t stride = -step_;
        const std::ptrdiff_t distance = start_ - stop_;
        return static_cast<std::size_t>((distance + stride - 1) / stride);
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return size() == 0;
    }

    [[nodiscard]] bool operator==(const Slice& other) const noexcept
    {
        return start_ == other.start_ && stop_ == other.stop_ && step_ == other.step_;
    }
};

}
