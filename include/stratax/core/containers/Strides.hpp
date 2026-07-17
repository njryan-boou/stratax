#pragma once

#include "Buffer.hpp"
#include "Shape.hpp"
#include "../Exceptions.hpp"
#include "../validation/Validation.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <ostream>

namespace stratax::core {

/**
 * @brief Stores strides for a shape in contiguous memory.
 *
 * Strides map logical indices to flat storage offsets, typically for
 * row-major layout. The values are stored as metadata and do not own
 * any array elements.
 */
class Strides
{
private:
    Buffer<std::size_t> buffer_;

public:
    /**
     * @brief Creates an empty stride vector.
     */
    Strides() noexcept = default;

    /**
     * @brief Creates row-major strides for a given shape.
     *
     * The last dimension receives stride 1 and each preceding dimension
     * is computed from the product of the dimensions to its right.
     *
     * @param shape Source shape used to derive stride values.
     *
     * @throws Exceptions::DimensionError If stride computation overflows.
     */
    explicit Strides(const Shape& shape)
        : buffer_(shape.rank())
    {
        if (shape.empty()) {
            return;
        }

        buffer_[shape.rank() - 1] = 1;

        for (std::size_t i = shape.rank() - 1; i > 0; --i) {
            buffer_[i - 1] = validation::checked_multiply(
                buffer_[i],
                shape(i),
                "Strides overflow for shape");
        }
    }

    Strides(const Strides&) = default;
    Strides(Strides&&) noexcept = default;

    Strides& operator=(const Strides&) = default;
    Strides& operator=(Strides&&) noexcept = default;

    /**
     * @brief Destroys the stride vector.
     */
    ~Strides() = default;

    /**
     * @brief Returns the number of stored stride values.
        *
        * @return Number of stored stride values.
     */
    [[nodiscard]] std::size_t size() const noexcept
    {
        return buffer_.size();
    }

    /**
     * @brief Returns whether the stride vector is empty.
        *
        * @return `true` when no strides are stored.
     */
    [[nodiscard]] bool empty() const noexcept
    {
        return buffer_.empty();
    }

    /**
     * @brief Returns the number of dimensions represented by the strides.
        *
        * @return Rank of the stride vector.
     */
    [[nodiscard]] std::size_t rank() const noexcept
    {
        return buffer_.size();
    }

    /**
     * @brief Returns a stride value without bounds checking.
     *
     * @param index Stride index.
     *
     * @return Const reference to the requested stride value.
     * @warning The caller must ensure that the index is in range.
     */
    const std::size_t& operator()(std::size_t index) const
    {
        return buffer_[index];
    }

    /**
     * @brief Returns a stride value with bounds checking.
     *
     * @param index Stride index.
     *
     * @return Const reference to the requested stride value.
     *
     * @throws Exceptions::IndexError If the index is out of bounds.
     */
    const std::size_t& at(std::size_t index) const
    {
        validation::require_index(index, rank(), "Strides index out of bounds");
        return buffer_[index];
    }

    /**
     * @brief Returns the first stride value.
        *
        * @return Const reference to the first stride value.
     */
    const std::size_t& front() const
    {
        return buffer_.front();
    }

    /**
     * @brief Returns the last stride value.
        *
        * @return Const reference to the last stride value.
     */
    const std::size_t& back() const
    {
        return buffer_.back();
    }

    /**
     * @brief Returns the raw stride data pointer.
        *
        * @return Pointer to the first stored stride, or `nullptr` when empty.
     */
    [[nodiscard]] const std::size_t* data() const noexcept
    {
        return buffer_.data();
    }

    /**
     * @brief Returns an iterator to the first stride value.
        *
        * @return Iterator to the beginning of the stride storage.
     */
    [[nodiscard]] const std::size_t* begin() noexcept
    {
        return buffer_.begin();
    }

    /**
     * @brief Returns a const iterator to the first stride value.
        *
        * @return Const iterator to the beginning of the stride storage.
     */
    [[nodiscard]] const std::size_t* begin() const noexcept
    {
        return buffer_.begin();
    }

    /**
     * @brief Returns a const iterator to the first stride value.
        *
        * @return Const iterator to the beginning of the stride storage.
     */
    [[nodiscard]] const std::size_t* cbegin() const noexcept
    {
        return buffer_.cbegin();
    }

    /**
     * @brief Returns an iterator one past the last stride value.
        *
        * @return Iterator to the end of the stride storage.
     */
    [[nodiscard]] const std::size_t* end() noexcept
    {
        return buffer_.end();
    }

    /**
     * @brief Returns a const iterator one past the last stride value.
        *
        * @return Const iterator to the end of the stride storage.
     */
    [[nodiscard]] const std::size_t* end() const noexcept
    {
        return buffer_.end();
    }

    /**
     * @brief Returns a const iterator one past the last stride value.
        *
        * @return Const iterator to the end of the stride storage.
     */
    [[nodiscard]] const std::size_t* cend() const noexcept
    {
        return buffer_.cend();
    }

    /**
     * @brief Returns a reverse iterator to the last stride value.
        *
        * @return Reverse iterator starting at the last stride value.
     */
    [[nodiscard]] std::reverse_iterator<const std::size_t*> rbegin() noexcept
    {
        return buffer_.rbegin();
    }

    /**
     * @brief Returns a const reverse iterator to the last stride value.
        *
        * @return Const reverse iterator starting at the last stride value.
     */
    [[nodiscard]] std::reverse_iterator<const std::size_t*> rbegin() const noexcept
    {
        return buffer_.rbegin();
    }

    /**
     * @brief Returns a const reverse iterator to the last stride value.
        *
        * @return Const reverse iterator starting at the last stride value.
     */
    [[nodiscard]] std::reverse_iterator<const std::size_t*> crbegin() const noexcept
    {
        return buffer_.crbegin();
    }

    /**
     * @brief Returns a reverse iterator before the first stride value.
        *
        * @return Reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<const std::size_t*> rend() noexcept
    {
        return buffer_.rend();
    }

    /**
     * @brief Returns a const reverse iterator before the first stride value.
        *
        * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<const std::size_t*> rend() const noexcept
    {
        return buffer_.rend();
    }

    /**
     * @brief Returns a const reverse iterator before the first stride value.
        *
        * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<const std::size_t*> crend() const noexcept
    {
        return buffer_.crend();
    }

    /**
     * @brief Swaps the stored strides with another instance.
        *
        * @param other Strides object to exchange state with.
     */
    void swap(Strides& other) noexcept
    {
        buffer_.swap(other.buffer_);
    }

    /**
     * @brief Compares two stride vectors for equality.
        *
        * @param other Strides object to compare against.
        *
        * @return `true` when the stride vectors match exactly.
     */
    [[nodiscard]] bool operator==(const Strides& other) const noexcept
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

    /**
     * @brief Returns whether two stride vectors differ.
        *
        * @param other Strides object to compare against.
        *
        * @return `true` when the stride vectors are not equal.
     */
    [[nodiscard]] bool operator!=(const Strides& other) const noexcept
    {
        return !(*this == other);
    }
};

/**
 * @brief Writes stride values in tuple-like form.
 *
 * @param os Output stream.
 * @param stride Stride vector to print.
 *
 * @return The updated output stream.
 */
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

