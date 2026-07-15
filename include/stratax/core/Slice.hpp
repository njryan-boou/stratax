#pragma once

#include "Exceptions.hpp"

#include <cstddef>

namespace stratax::core {

/**
 * @brief Represents a half-open strided range of indices.
 *
 * Slice stores an inclusive start, exclusive stop, and non-zero step.
 * It is used by slicing helpers to describe one-dimensional index intervals.
 */
class Slice
{
private:
    std::ptrdiff_t start_;
    std::ptrdiff_t stop_;
    std::ptrdiff_t step_;

public:
    /**
    * @brief Creates a half-open strided slice range.
     *
     * A slice includes the start index and excludes the stop index.
     *
     * @param start First index included in the slice.
     * @param stop Index one past the last included element.
    * @param step Non-zero stride between selected indices.
     *
     * @throws Exceptions::IndexError If step is zero.
     */
    Slice(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step = 1)
        : start_(start),
          stop_(stop),
          step_(step)
    {
        if (step == 0) {
            throw Exceptions::IndexError("Slice step cannot be zero.");
        }
    }

    /**
     * @brief Returns the first index in the slice.
        *
        * @return The inclusive start index.
     */
    [[nodiscard]] std::ptrdiff_t start() const noexcept
    {
        return start_;
    }

    /**
     * @brief Returns the index one past the end of the slice.
        *
        * @return The exclusive stop index.
     */
    [[nodiscard]] std::ptrdiff_t stop() const noexcept
    {
        return stop_;
    }

    /**
     * @brief Returns the stride between selected indices.
        *
        * @return Positive slice step.
     */
    [[nodiscard]] std::ptrdiff_t step() const noexcept
    {
        return step_;
    }

    /**
     * @brief Returns the number of indices covered by the slice.
        *
        * @return Number of selected indices.
     */
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

    /**
     * @brief Returns whether the slice selects no elements.
        *
        * @return `true` when the slice is empty.
     */
    [[nodiscard]] bool empty() const noexcept
    {
        return size() == 0;
    }

    /**
    * @brief Compares two slices for identical bounds and step.
     *
     * @param other Slice to compare against.
     *
     * @return `true` when start, stop, and step all match.
     */
    [[nodiscard]] bool operator==(const Slice& other) const noexcept
    {
        return start_ == other.start_ && stop_ == other.stop_ && step_ == other.step_;
    }

    /**
     * @brief Returns whether two slices have different bounds.
     *
     * @param other Slice to compare against.
     *
     * @return `true` when the slices are not equal.
     */
    [[nodiscard]] bool operator!=(const Slice& other) const noexcept
    {
        return !(*this == other);
    }
};

}
