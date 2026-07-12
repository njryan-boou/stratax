#pragma once

#include "Exceptions.hpp"

#include <cstddef>

namespace stratax::core {

/**
 * @brief Represents a half-open range of indices.
 *
 * Slice stores an inclusive start and exclusive stop bound and is used
 * by slicing helpers to describe one-dimensional index intervals.
 */
class Slice
{
private:
    std::size_t start_;
    std::size_t stop_;

public:
    /**
     * @brief Creates a half-open slice range.
     *
     * A slice includes the start index and excludes the stop index.
     *
     * @param start First index included in the slice.
     * @param stop Index one past the last included element.
     *
     * @throws Exceptions::IndexError If start is greater than stop.
     */
    Slice(std::size_t start, std::size_t stop)
        : start_(start),
          stop_(stop)
    {
        if (start > stop) {
            throw Exceptions::IndexError("Slice start cannot be greater than stop.");
        }
    }

    /**
     * @brief Returns the first index in the slice.
        *
        * @return The inclusive start index.
     */
    [[nodiscard]] std::size_t start() const noexcept
    {
        return start_;
    }

    /**
     * @brief Returns the index one past the end of the slice.
        *
        * @return The exclusive stop index.
     */
    [[nodiscard]] std::size_t stop() const noexcept
    {
        return stop_;
    }

    /**
     * @brief Returns the number of indices covered by the slice.
        *
        * @return Number of selected indices.
     */
    [[nodiscard]] std::size_t size() const noexcept
    {
        return stop_ - start_;
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
     * @brief Compares two slices for identical bounds.
     *
     * @param other Slice to compare against.
     *
     * @return `true` when both the start and stop bounds match.
     */
    [[nodiscard]] bool operator==(const Slice& other) const noexcept
    {
        return start_ == other.start_ && stop_ == other.stop_;
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
