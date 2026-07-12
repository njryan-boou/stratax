#pragma once

#include <cstddef>
#include <initializer_list>
#include <iterator>

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Validation.hpp>

namespace stratax::container {

template<typename T>
requires Numeric<T>
/**
 * @brief Stores a rank-1 Stratax array in contiguous memory.
 *
 * Vector owns its element storage and pairs it with shape and stride
 * metadata so the container can participate in the generic array API.
 *
 * @tparam T Numeric element type.
 */
class Vector
{
private:
    core::Shape shape_;
    core::Strides strides_;
    core::Buffer<T> buffer_;

public:
    using value_type = T;

    template<typename U>
    using rebind = Vector<U>;

    /**
     * @brief Creates a default vector with no logical dimensions.
     *
     * The default vector has rank 0 and owns no element storage. Use
     * Vector(0) when a rank-1 empty vector is required.
     *
     */
    Vector() noexcept = default;

    /**
     * @brief Creates a vector with the given number of elements.
     *
     * @param size Number of elements to allocate.
     *
     */
    explicit Vector(std::size_t size)
        : shape_({size}, core::Shape::allow_zero),
          strides_(shape_),
          buffer_(size)
    {
    }

    /**
     * @brief Creates a vector from a validated rank-1 shape.
     *
     * @param shape Source shape.
     *
     * @throws Exceptions::DimensionError If the shape is not rank 1.
     */
    explicit Vector(const core::Shape& shape)
        : shape_(core::validation::require_rank(shape, 1, "Shape must be rank 1")),
          strides_(shape_),
          buffer_(shape_.elements())
    {
    }

    /**
     * @brief Creates a vector and fills it with a value.
     *
     * @param size Number of elements to allocate.
     * @param value Value to copy into each element.
     *
     */
    Vector(std::size_t size, const T& value)
        : shape_({size}, core::Shape::allow_zero),
          strides_(shape_),
          buffer_(size, value)
    {
    }

    /**
     * @brief Creates a vector from an initializer list.
     *
     * @param list Source values in element order.
     *
     */
    Vector(std::initializer_list<T> list)
        : shape_({list.size()}, core::Shape::allow_zero),
          strides_(shape_),
          buffer_(list)
    {
    }

    // Rule of Five

    /**
     * @brief Creates a copy of another vector.
     *
     */
    Vector(const Vector&) = default;

    /**
     * @brief Transfers ownership from another vector.
     *
     */
    Vector(Vector&&) noexcept = default;

    /**
     * @brief Replaces this vector with a copy of another vector.
     *
     */
    Vector& operator=(const Vector&) = default;

    /**
     * @brief Replaces this vector by taking ownership from another vector.
     *
     */
    Vector& operator=(Vector&&) noexcept = default;

    /**
     * @brief Destroys the vector.
     *
     */
    ~Vector() = default;

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
     * @brief Returns a flat element with bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Mutable reference to the indexed element.
     *
     * @throws Exceptions::IndexError If the index is out of bounds.
     */
    T& at(std::size_t index)
    {
        core::validation::require_index(index, size(), "Vector index out of bounds.");
        return buffer_[index];
    }

    /**
     * @brief Returns a flat element with bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Const reference to the indexed element.
     *
     * @throws Exceptions::IndexError If the index is out of bounds.
     */
    const T& at(std::size_t index) const
    {
        core::validation::require_index(index, size(), "Vector index out of bounds.");
        return buffer_[index];
    }

    /**
     * @brief Returns the first element.
        *
        * @return Mutable reference to the first stored element.
     */
    T& front()
    {
        return buffer_.front();
    }

    /**
     * @brief Returns the first element as a const reference.
        *
        * @return Const reference to the first stored element.
     */
    const T& front() const
    {
        return buffer_.front();
    }

    /**
     * @brief Returns the last element.
        *
        * @return Mutable reference to the last stored element.
     */
    T& back()
    {
        return buffer_.back();
    }

    /**
     * @brief Returns the last element as a const reference.
        *
        * @return Const reference to the last stored element.
     */
    const T& back() const
    {
        return buffer_.back();
    }

    /**
     * @brief Returns the number of stored elements.
        *
        * @return Number of elements in the vector.
     */
    [[nodiscard]] std::size_t size() const noexcept
    {
    return shape_.elements();
    }

    /**
     * @brief Returns the vector rank.
     *
     * @return The number of dimensions, normally 1 for constructed vectors and 0 for the default state.
     */
    [[nodiscard]] std::size_t rank() const noexcept
    {
        return shape_.rank();
    }

    /**
     * @brief Returns whether the vector has no elements.
        *
        * @return `true` when the vector size is zero.
     */
    [[nodiscard]] bool empty() const noexcept
    {
    return buffer_.empty();
    }

    /**
     * @brief Returns the vector shape.
        *
        * @return Shape describing the vector.
     */
    const stratax::core::Shape& shape() const noexcept
    {
    return shape_;
    }

    /**
     * @brief Returns the vector strides.
        *
        * @return Strides describing the vector.
     */
    const stratax::core::Strides& strides() const noexcept
    {
        return strides_;
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
        * @return Iterator to the beginning of the vector.
     */
    [[nodiscard]] T* begin() noexcept
    {
        return buffer_.begin();
    }

    /**
     * @brief Returns a const iterator to the first element.
        *
        * @return Const iterator to the beginning of the vector.
     */
    [[nodiscard]] const T* begin() const noexcept
    {
        return buffer_.begin();
    }

    /**
     * @brief Returns a const iterator to the first element.
        *
        * @return Const iterator to the beginning of the vector.
     */
    [[nodiscard]] const T* cbegin() const noexcept
    {
        return buffer_.cbegin();
    }

    /**
     * @brief Returns an iterator one past the last element.
        *
        * @return Iterator to the end of the vector.
     */
    [[nodiscard]] T* end() noexcept
    {
        return buffer_.end();
    }

    /**
     * @brief Returns a const iterator one past the last element.
        *
        * @return Const iterator to the end of the vector.
     */
    [[nodiscard]] const T* end() const noexcept
    {
        return buffer_.end();
    }

    /**
     * @brief Returns a const iterator one past the last element.
        *
        * @return Const iterator to the end of the vector.
     */
    [[nodiscard]] const T* cend() const noexcept
    {
        return buffer_.cend();
    }

    /**
     * @brief Returns a reverse iterator to the last element.
        *
        * @return Reverse iterator starting at the last element.
     */
    [[nodiscard]] std::reverse_iterator<T*> rbegin() noexcept
    {
    return buffer_.rbegin();
    }

    /**
     * @brief Returns a const reverse iterator to the last element.
        *
        * @return Const reverse iterator starting at the last element.
     */
    [[nodiscard]] std::reverse_iterator<const T*> rbegin() const noexcept
    {
        return buffer_.rbegin();
    }

    /**
     * @brief Returns a const reverse iterator to the last element.
        *
        * @return Const reverse iterator starting at the last element.
     */
    [[nodiscard]] std::reverse_iterator<const T*> crbegin() const noexcept
    {
        return buffer_.crbegin();
    }

    /**
     * @brief Returns a reverse iterator before the first element.
        *
        * @return Reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<T*> rend() noexcept
    {
        return buffer_.rend();
    }

    /**
     * @brief Returns a const reverse iterator before the first element.
        *
        * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<const T*> rend() const noexcept
    {
        return buffer_.rend();
    }

    /**
     * @brief Returns a const reverse iterator before the first element.
        *
        * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<const T*> crend() const noexcept
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
     * @brief Swaps the contents of two vectors.
     *
     * @param other Vector to exchange state with.
     */
    void swap(Vector& other) noexcept
    {
        using std::swap;

        swap(shape_, other.shape_);
        swap(strides_, other.strides_);
        swap(buffer_, other.buffer_);
    }

};

}
