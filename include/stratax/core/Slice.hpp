#pragma once

#include "Exceptions.hpp"

#include <cstddef>

namespace stratax::core {

/** @brief Represents a half-open strided range of indices. */
class Slice
{
private:
    std::ptrdiff_t start_;
    std::ptrdiff_t stop_;
    std::ptrdiff_t step_;

public:
    /** @brief Creates a half-open strided slice range with non-zero step. */
    Slice(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step = 1)
        : start_(start),
          stop_(stop),
          step_(step)
    {
        if (step == 0) {
            throw Exceptions::IndexError("Slice step cannot be zero.");
        }
    }

    /** @brief Returns the inclusive start index. */
    [[nodiscard]] std::ptrdiff_t start() const noexcept
    {
        return start_;
    }

    /** @brief Returns the exclusive stop index. */
    [[nodiscard]] std::ptrdiff_t stop() const noexcept
    {
        return stop_;
    }

    /** @brief Returns the stride between selected indices. */
    [[nodiscard]] std::ptrdiff_t step() const noexcept
    {
        return step_;
    }

    /** @brief Returns the number of indices covered by the slice. */
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

    /** @brief Returns whether the slice selects no elements. */
    [[nodiscard]] bool empty() const noexcept
    {
        return size() == 0;
    }

    /** @brief Compares two slices for identical bounds and step. */
    [[nodiscard]] bool operator==(const Slice& other) const noexcept
    {
        return start_ == other.start_ && stop_ == other.stop_ && step_ == other.step_;
    }

    /** @brief Returns whether two slices have different bounds. */
    [[nodiscard]] bool operator!=(const Slice& other) const noexcept
    {
        return !(*this == other);
    }
};

}
