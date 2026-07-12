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
/**
 * @brief Stores an N-dimensional Stratax array in contiguous memory.
 *
 * Tensor owns element storage and shape metadata for generic
 * multi-dimensional array operations.
 *
 * @tparam T Numeric element type.
 */
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

    /**
     * @brief Creates an empty tensor with rank 0.
     *
     * The tensor has no dimensions, no strides, and owns no elements.
     *
     */
    Tensor() noexcept = default;

    /**
     * @brief Creates a tensor with default-initialized storage for a shape.
     *
     * The tensor rank and strides are derived directly from `shape`.
     *
     * @param shape Source shape.
     *
     */
    explicit Tensor(const core::Shape& shape)
        : shape_(shape),
          strides_(shape_),
          buffer_(shape_.elements())
    {
    }

    /**
     * @brief Creates a tensor from a shape and fills it with a value.
     *
     * @param shape Source shape.
     * @param value Value copied into every element.
     *
     */
    Tensor(const core::Shape& shape, const T& value)
        : shape_(shape),
          strides_(shape_),
          buffer_(shape_.elements(), value)
    {
    }

    /**
     * @brief Creates a copy of another tensor.
     *
     */
    Tensor(const Tensor&) = default;

    /**
     * @brief Transfers ownership from another tensor.
     *
     */
    Tensor(Tensor&&) noexcept = default;

    /**
     * @brief Replaces this tensor with a copy of another tensor.
     *
     */
    Tensor& operator=(const Tensor&) = default;

    /**
     * @brief Replaces this tensor by taking ownership from another tensor.
     *
     */
    Tensor& operator=(Tensor&&) noexcept = default;

    /**
     * @brief Destroys the tensor.
     *
     */
    ~Tensor() = default;

    /**
     * @brief Returns the total number of elements in the tensor.
     *
     * @return Number of stored elements.
     */
    [[nodiscard]] std::size_t size() const noexcept
    {
        return buffer_.size();
    }

    /**
     * @brief Returns whether the tensor contains no elements.
     *
     * @return `true` when the tensor size is zero.
     */
    [[nodiscard]] bool empty() const noexcept
    {
        return buffer_.empty();
    }

    /**
     * @brief Returns the tensor rank.
     *
     * @return Number of dimensions described by the tensor shape.
     */
    [[nodiscard]] std::size_t rank() const noexcept
    {
        return shape_.rank();
    }

    /**
     * @brief Returns the tensor shape.
     *
     * @return Shape describing the tensor.
     */
    const core::Shape& shape() const noexcept
    {
        return shape_;
    }

    /**
     * @brief Returns the tensor strides.
     *
     * @return Row-major strides describing the tensor layout.
     */
    const core::Strides& strides() const noexcept
    {
        return strides_;
    }

    /**
     * @brief Returns a flat element without bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Mutable reference to the indexed element.
     * @warning The caller must ensure that the index is in range.
     */
    T& operator()(std::size_t index) noexcept
    {
        return buffer_[index];
    }

    /**
     * @brief Returns a flat element without bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Const reference to the indexed element.
     * @warning The caller must ensure that the index is in range.
     */
    const T& operator()(std::size_t index) const noexcept
    {
        return buffer_[index];
    }

    /**
     * @brief Returns a flat element without bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Mutable reference to the indexed element.
     * @warning The caller must ensure that the index is in range.
     */
    T& operator[](std::size_t index) noexcept
    {
        return buffer_[index];
    }

    /**
     * @brief Returns a flat element without bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Const reference to the indexed element.
     * @warning The caller must ensure that the index is in range.
     */
    const T& operator[](std::size_t index) const noexcept
    {
        return buffer_[index];
    }

    /**
     * @brief Returns an element by multidimensional index with bounds checking.
     *
     * At least two index components are required; use the flat overload for
     * rank-1 or flat storage access.
     *
     * @param first First index component.
     * @param second Second index component.
     * @param rest Remaining index components.
     *
     * @return Mutable reference to the indexed element.
     * @throws Exceptions::DimensionError If the number of indices does not match the tensor rank.
     * @throws Exceptions::IndexError If any index component is out of bounds or the offset overflows.
     */
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

    /**
     * @brief Returns an element by multidimensional index with bounds checking.
     *
     * @param first First index component.
     * @param second Second index component.
     * @param rest Remaining index components.
     *
     * @return Const reference to the indexed element.
     * @throws Exceptions::DimensionError If the number of indices does not match the tensor rank.
     * @throws Exceptions::IndexError If any index component is out of bounds or the offset overflows.
     */
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

    /**
     * @brief Returns an element by rank-1 flat index with bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Mutable reference to the indexed element.
     * @throws Exceptions::IndexError If the index is out of bounds.
     */
    T& at(std::size_t index)
    {
        core::validation::require_index(index, size(), "Tensor flat index out of bounds.");
        return buffer_[index];
    }

    /**
     * @brief Returns an element by rank-1 flat index with bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Const reference to the indexed element.
     * @throws Exceptions::IndexError If the index is out of bounds.
     */
    const T& at(std::size_t index) const
    {
        core::validation::require_index(index, size(), "Tensor flat index out of bounds.");
        return buffer_[index];
    }

    /**
     * @brief Returns an element by multi-index with bounds checking.
     *
     * @param first First index component.
     * @param second Second index component.
     * @param rest Remaining index components.
     *
     * @return Mutable reference to the indexed element.
     *
     * @throws Exceptions::IndexError If the index rank or bounds are invalid.
     */
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
            throw Exceptions::IndexError("Tensor multi-index rank must match tensor rank.");
        }
        catch (const Exceptions::IndexError&)
        {
            throw Exceptions::IndexError("Tensor multi-index component is out of bounds.");
        }
    }

    /**
     * @brief Returns an element by multi-index with bounds checking.
     *
     * @param first First index component.
     * @param second Second index component.
     * @param rest Remaining index components.
     *
     * @return Const reference to the indexed element.
     *
     * @throws Exceptions::IndexError If the index rank or bounds are invalid.
     */
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
            throw Exceptions::IndexError("Tensor multi-index rank must match tensor rank.");
        }
        catch (const Exceptions::IndexError&)
        {
            throw Exceptions::IndexError("Tensor multi-index component is out of bounds.");
        }
    }

    /**
     * @brief Returns the first element.
     *
     * @return Mutable reference to the first stored element.
     * @throws Exceptions::IndexError If the tensor is empty.
     */
    T& front()
    {
        return buffer_.front();
    }

    /**
     * @brief Returns the first element as a const reference.
     *
     * @return Const reference to the first stored element.
     * @throws Exceptions::IndexError If the tensor is empty.
     */
    const T& front() const
    {
        return buffer_.front();
    }

    /**
     * @brief Returns the last element.
     *
     * @return Mutable reference to the last stored element.
     * @throws Exceptions::IndexError If the tensor is empty.
     */
    T& back()
    {
        return buffer_.back();
    }

    /**
     * @brief Returns the last element as a const reference.
     *
     * @return Const reference to the last stored element.
     * @throws Exceptions::IndexError If the tensor is empty.
     */
    const T& back() const
    {
        return buffer_.back();
    }

    /**
     * @brief Returns the raw data pointer.
     *
     * @return Pointer to the first stored element, or `nullptr` when empty.
     */
    [[nodiscard]] T* data() noexcept
    {
        return buffer_.data();
    }

    /**
     * @brief Returns the raw data pointer as a const pointer.
     *
     * @return Pointer to the first stored element, or `nullptr` when empty.
     */
    [[nodiscard]] const T* data() const noexcept
    {
        return buffer_.data();
    }

    /**
     * @brief Returns an iterator to the first element.
     *
     * @return Iterator to the beginning of the tensor.
     */
    [[nodiscard]] auto begin() noexcept
    {
        return buffer_.begin();
    }

    /**
     * @brief Returns a const iterator to the first element.
     *
     * @return Const iterator to the beginning of the tensor.
     */
    [[nodiscard]] auto begin() const noexcept
    {
        return buffer_.begin();
    }

    /**
     * @brief Returns a const iterator to the first element.
     *
     * @return Const iterator to the beginning of the tensor.
     */
    [[nodiscard]] auto cbegin() const noexcept
    {
        return buffer_.cbegin();
    }

    /**
     * @brief Returns an iterator one past the last element.
     *
     * @return Iterator to the end of the tensor.
     */
    [[nodiscard]] auto end() noexcept
    {
        return buffer_.end();
    }

    /**
     * @brief Returns a const iterator one past the last element.
     *
     * @return Const iterator to the end of the tensor.
     */
    [[nodiscard]] auto end() const noexcept
    {
        return buffer_.end();
    }

    /**
     * @brief Returns a const iterator one past the last element.
     *
     * @return Const iterator to the end of the tensor.
     */
    [[nodiscard]] auto cend() const noexcept
    {
        return buffer_.cend();
    }

    /**
     * @brief Returns a reverse iterator to the last element.
     *
     * @return Reverse iterator starting at the last element.
     */
    [[nodiscard]] auto rbegin() noexcept
    {
        return buffer_.rbegin();
    }

    /**
     * @brief Returns a const reverse iterator to the last element.
     *
     * @return Const reverse iterator starting at the last element.
     */
    [[nodiscard]] auto rbegin() const noexcept
    {
        return buffer_.rbegin();
    }

    /**
     * @brief Returns a const reverse iterator to the last element.
     *
     * @return Const reverse iterator starting at the last element.
     */
    [[nodiscard]] auto crbegin() const noexcept
    {
        return buffer_.crbegin();
    }

    /**
     * @brief Returns a reverse iterator before the first element.
     *
     * @return Reverse iterator representing the end sentinel.
     */
    [[nodiscard]] auto rend() noexcept
    {
        return buffer_.rend();
    }

    /**
     * @brief Returns a const reverse iterator before the first element.
     *
     * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] auto rend() const noexcept
    {
        return buffer_.rend();
    }

    /**
     * @brief Returns a const reverse iterator before the first element.
     *
     * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] auto crend() const noexcept
    {
        return buffer_.crend();
    }

    /**
     * @brief Fills every element with the same value.
     *
     * @param value Value to assign to each element.
     */
    void fill(const T& value)
    {
        buffer_.fill(value);
    }

    /**
     * @brief Swaps the contents of two tensors.
     *
     * @param other Tensor to exchange state with.
     */
    void swap(Tensor& other) noexcept
    {
        shape_.swap(other.shape_);
        strides_.swap(other.strides_);
        buffer_.swap(other.buffer_);
    }
};

}
