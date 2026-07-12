#pragma once

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <utility>
#include <new>
#include <limits>
#include <type_traits>

#include "Config.hpp"
#include "Exceptions.hpp"

namespace stratax::core {

/**
 * @brief Owns contiguous dynamically allocated storage.
 *
 * Buffer is the low-level memory container used by Stratax arrays.
 * It manages allocation, destruction, and flat element access, while
 * higher-level types such as Vector, Matrix, and Tensor handle shape
 * semantics and mathematical operations.
 *
 * The stored elements are always contiguous in memory, and the buffer
 * follows RAII so ownership is released automatically when the object
 * is destroyed.
 *
 * @tparam T Element type.
 * @tparam Alignment Alignment used for allocation, defaulting to 64.
 *
 * @note Buffer provides storage only. It does not validate shapes or
 * perform multidimensional indexing.
 */
template<typename T, std::size_t Alignment = config::default_alignment>
class Buffer {
public:
    struct uninitialized_t {};
    static constexpr uninitialized_t uninitialized{};

    /**
     * @brief Creates an empty buffer.
     *
     * The buffer owns no storage and contains zero elements.
     */
    Buffer() noexcept: data_(nullptr), size_(0) {}

    /**
     * @brief Creates a buffer with default-initialized elements.
     *
     * The buffer allocates storage for the requested number of elements
     * and default-constructs each one in place.
     *
     * @param size Number of elements to allocate.
     */
    explicit Buffer(std::size_t size): data_(allocate(size)), size_(size)
    {
        construct_default(0, size_);
    }

    /**
     * @brief Creates a buffer with allocated but uninitialized storage.
     *
     * Use this overload when you intend to construct the elements later
     * and want to avoid default initialization work.
     *
     * @param size Number of elements to allocate.
     * @param uninitialized Tag selecting uninitialized storage.
     *
     * @warning The element type must be trivially destructible.
     */
    Buffer(std::size_t size, uninitialized_t uninitialized): data_(allocate(size)), size_(size)
    {
        (void)uninitialized;
        static_assert(
            std::is_trivially_destructible_v<T>,
            "uninitialized Buffer storage is only safe for trivially destructible types"
        );
    }

    /**
     * @brief Creates a buffer and fills every element with a value.
     *
     * Each element is copy-constructed from the supplied value.
     *
     * @param size Number of elements to allocate.
     * @param value Value to copy into each element.
     */
    Buffer(std::size_t size, const T& value): data_(allocate(size)), size_(size)
    {
        construct_fill(0, size_, value);
    }

    /**
     * @brief Creates a buffer from an initializer list.
     *
     * The elements are copied in list order.
     *
     * @param list Source values to copy into the buffer.
     */
    Buffer(std::initializer_list<T> list): data_(allocate(list.size())), size_(list.size())
    {
        std::size_t i = 0;

        try {
            for (const T& value : list) {
                std::construct_at(data_ + i, value);
                ++i;
            }
        } catch (...) {
            destroy(0, i);
            deallocate(data_);
            throw;
        }
    }

    /**
     * @brief Creates a copy of another buffer.
     *
     * The new buffer owns its own storage and duplicates the source
     * elements.
     *
     * @param other Buffer to copy from.
     */
    Buffer(const Buffer& other)
        : data_(allocate(other.size_)), size_(other.size_)
    {
        if constexpr (std::is_trivially_copyable_v<T>) {
            if (size_ != 0) {
                std::memcpy(data_, other.data_, sizeof(T) * size_);
            }
            return;
        }

        std::size_t i = 0;

        try {
            for (; i < size_; ++i) {
                std::construct_at(data_ + i, other.data_[i]);
            }
        } catch (...) {
            destroy(0, i);
            deallocate(data_);
            throw;
        }
    }

    /**
     * @brief Transfers ownership from another buffer.
     *
     * The source buffer is left empty after the move.
     *
     * @param other Buffer to move from.
     */
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief Replaces this buffer with a copy of another buffer.
     *
     * This assignment provides the strong exception guarantee by
     * copying into a temporary buffer first.
     *
     * @param other Buffer to copy from.
     *
     * @return Reference to this buffer.
     */
    Buffer& operator=(const Buffer& other)
    {
        if (this == &other)
            return *this;

        Buffer temp(other);
        swap(temp);
        return *this;
    }

    /**
     * @brief Replaces this buffer by taking ownership from another buffer.
     *
     * The current contents are destroyed before the source storage is
     * adopted.
     *
     * @param other Buffer to move from.
     *
     * @return Reference to this buffer.
     */
    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this == &other)
            return *this;

        destroy(0, size_);
        deallocate(data_);

        data_ = other.data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;

        return *this;
    }

    /**
     * @brief Releases the owned storage.
     *
     * Destroying a buffer also destroys any constructed elements before
     * the memory is returned to the allocator.
     */
    ~Buffer()
    {
        destroy(0, size_);
        deallocate(data_);
    }

    /**
     * @brief Returns a mutable element reference without bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Reference to the indexed element.
     *
     * @warning The caller must ensure that the index is in range.
     */
    T& operator[](std::size_t index) noexcept
    {
        return data_[index];
    }

    /**
     * @brief Returns a const element reference without bounds checking.
     *
     * @param index Flat element index.
     *
     * @return Const reference to the indexed element.
     *
     * @warning The caller must ensure that the index is in range.
     */
    const T& operator[](std::size_t index) const noexcept
    {
        return data_[index];
    }

    /**
     * @brief Returns the first element.
     *
     * @return Mutable reference to the first stored element.
     *
     * @throws Exceptions::IndexError If the buffer is empty.
     */
    T& front()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    /**
     * @brief Returns the first element as a const reference.
     *
     * @return Const reference to the first stored element.
     *
     * @throws Exceptions::IndexError If the buffer is empty.
     */
    const T& front() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    /**
     * @brief Returns the last element.
     *
     * @return Mutable reference to the last stored element.
     *
     * @throws Exceptions::IndexError If the buffer is empty.
     */
    T& back()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    /**
     * @brief Returns the last element as a const reference.
     *
     * @return Const reference to the last stored element.
     *
     * @throws Exceptions::IndexError If the buffer is empty.
     */
    const T& back() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    /**
     * @brief Returns the raw data pointer.
        *
        * @return Pointer to the first stored element, or `nullptr` when empty.
     */
    [[nodiscard]] T* data() noexcept
    {
        return data_;
    }

    /**
     * @brief Returns the raw data pointer as a const pointer.
        *
        * @return Pointer to the first stored element, or `nullptr` when empty.
     */
    [[nodiscard]] const T* data() const noexcept
    {
        return data_;
    }

    /**
     * @brief Returns an iterator to the first element.
        *
        * @return Iterator to the beginning of the buffer.
     */
    [[nodiscard]] T* begin() noexcept
    {
        return data_;
    }

    /**
     * @brief Returns a const iterator to the first element.
        *
        * @return Const iterator to the beginning of the buffer.
     */
    [[nodiscard]] const T* begin() const noexcept
    {
        return data_;
    }

    /**
     * @brief Returns a const iterator to the first element.
        *
        * @return Const iterator to the beginning of the buffer.
     */
    [[nodiscard]] const T* cbegin() const noexcept
    {
        return data_;
    }

    /**
     * @brief Returns an iterator one past the last element.
        *
        * @return Iterator to the end of the buffer.
     */
    [[nodiscard]] T* end() noexcept
    {
        return data_ + size_;
    }

    /**
     * @brief Returns a const iterator one past the last element.
        *
        * @return Const iterator to the end of the buffer.
     */
    [[nodiscard]] const T* end() const noexcept
    {
        return data_ + size_;
    }

    /**
     * @brief Returns a const iterator one past the last element.
        *
        * @return Const iterator to the end of the buffer.
     */
    [[nodiscard]] const T* cend() const noexcept
    {
        return data_ + size_;
    }

    /**
     * @brief Returns a reverse iterator to the last element.
        *
        * @return Reverse iterator starting at the last element.
     */
    [[nodiscard]] std::reverse_iterator<T*> rbegin() noexcept
    {
        return std::reverse_iterator<T*>(end());
    }

    /**
     * @brief Returns a const reverse iterator to the last element.
        *
        * @return Const reverse iterator starting at the last element.
     */
    [[nodiscard]] std::reverse_iterator<const T*> rbegin() const noexcept
    {
        return std::reverse_iterator<const T*>(end());
    }

    /**
     * @brief Returns a const reverse iterator to the last element.
        *
        * @return Const reverse iterator starting at the last element.
     */
    [[nodiscard]] std::reverse_iterator<const T*> crbegin() const noexcept
    {
        return std::reverse_iterator<const T*>(cend());
    }

    /**
     * @brief Returns a reverse iterator before the first element.
        *
        * @return Reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<T*> rend() noexcept
    {
        return std::reverse_iterator<T*>(begin());
    }

    /**
     * @brief Returns a const reverse iterator before the first element.
        *
        * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<const T*> rend() const noexcept
    {
        return std::reverse_iterator<const T*>(begin());
    }

    /**
     * @brief Returns a const reverse iterator before the first element.
        *
        * @return Const reverse iterator representing the end sentinel.
     */
    [[nodiscard]] std::reverse_iterator<const T*> crend() const noexcept
    {
        return std::reverse_iterator<const T*>(cbegin());
    }

    /**
     * @brief Returns the number of stored elements.
        *
        * @return Number of elements currently owned by the buffer.
     */
    [[nodiscard]] std::size_t size() const noexcept
    {
        return size_;
    }

    /**
     * @brief Returns whether the buffer contains no elements.
        *
        * @return `true` when the buffer is empty; otherwise `false`.
     */
    [[nodiscard]] bool empty() const noexcept
    {
        return size_ == 0;
    }

    /**
     * @brief Fills every element with the same value.
     *
     * @param value Value to assign to each element.
     */
    void fill(const T& value)
    {
        std::fill_n(data_, size_, value);
    }

    /**
     * @brief Swaps the contents of two buffers.
     *
     * @param other Buffer to exchange state with.
     */
    void swap(Buffer& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

private:
    T* data_;
    std::size_t size_;

private:
    /**
     * @brief Allocates aligned raw storage for a number of elements.
     *
     * @param space Number of elements to allocate.
     *
     * @return Pointer to uninitialized aligned storage.
     *
     * @throws std::bad_array_new_length If the requested storage would overflow.
     */
    static T* allocate(std::size_t space)
    {
        if (space == 0)
            return nullptr;

        if (space > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        return static_cast<T*>(
            ::operator new(
                sizeof(T) * space,
                std::align_val_t{Alignment}
            )
        );
    }

    /**
     * @brief Releases aligned raw storage previously allocated by the buffer.
     *
     * @param ptr Pointer returned by allocate.
     */
    static void deallocate(T* ptr) noexcept
    {
        if (ptr == nullptr)
            return;

        ::operator delete(
            ptr,
            std::align_val_t{Alignment}
        );
    }

    /**
     * @brief Fills a range of elements by copy-constructing each value in-place.
     *
     * @param begin First index in the construction range.
     * @param end One past the last index in the construction range.
     * @param value Value to copy into each constructed element.
     *
     * @throws Any exception thrown by element construction is propagated after cleanup.
     */
    void construct_fill(std::size_t begin, std::size_t end, const T& value)
    {
        try {
            std::uninitialized_fill_n(data_ + begin, end - begin, value);
        } catch (...) {
            deallocate(data_);
            data_ = nullptr;
            size_ = 0;
            throw;
        }
    }

    /**
     * @brief Value-constructs a range of elements in-place.
     *
     * @param begin First index in the construction range.
     * @param end One past the last index in the construction range.
     *
     * @throws Any exception thrown by element construction is propagated after cleanup.
     */
    void construct_default(std::size_t begin, std::size_t end)
    {
        try {
            std::uninitialized_value_construct_n(data_ + begin, end - begin);
        } catch (...) {
            deallocate(data_);
            data_ = nullptr;
            size_ = 0;
            throw;
        }
    }

    /**
     * @brief Destroys a range of constructed elements in-place.
     *
     * @param begin First index in the destruction range.
     * @param end One past the last index in the destruction range.
     */
    void destroy(std::size_t begin, std::size_t end) noexcept
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (std::size_t index = begin; index < end; ++index) {
                std::destroy_at(data_ + index);
            }
        }
    }
};

}
