#pragma once

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "../Config.hpp"
#include "../Exceptions.hpp"

namespace stratax::core {

/** @brief Owns contiguous dynamically allocated storage for Stratax arrays. */
template<typename T, std::size_t Alignment = config::default_alignment>
class Buffer {
public:
    /** @brief Tag type selecting allocated storage without element initialization. */
    struct uninitialized_t {};

    /** @brief Tag value selecting allocated storage without element initialization. */
    static constexpr uninitialized_t uninitialized{};

    /** @brief Mutable iterator over contiguous buffer elements. */
    using iterator = T*;

    /** @brief Const iterator over contiguous buffer elements. */
    using const_iterator = const T*;

    /** @brief Mutable reverse iterator over contiguous buffer elements. */
    using reverse_iterator = std::reverse_iterator<iterator>;

    /** @brief Const reverse iterator over contiguous buffer elements. */
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /** @brief Creates an empty buffer. */
    Buffer() noexcept : data_(nullptr), size_(0) {}

    /** @brief Creates a buffer with default-initialized elements. */
    explicit Buffer(std::size_t size) : data_(allocate(size)), size_(size)
    {
        construct_default(0, size_);
    }

    /** @brief Creates a buffer with allocated but uninitialized storage. */
    Buffer(std::size_t size, uninitialized_t uninitialized) : data_(allocate(size)), size_(size)
    {
        (void)uninitialized;
        static_assert(
            std::is_trivially_destructible_v<T>,
            "uninitialized Buffer storage is only safe for trivially destructible types"
        );
    }

    /** @brief Creates a buffer and fills every element with a value. */
    Buffer(std::size_t size, const T& value) : data_(allocate(size)), size_(size)
    {
        construct_fill(0, size_, value);
    }

    /** @brief Creates a buffer from an initializer list. */
    Buffer(std::initializer_list<T> list) : data_(allocate(list.size())), size_(list.size())
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

    /** @brief Creates a copy of another buffer. */
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

    /** @brief Transfers ownership from another buffer. */
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /** @brief Replaces this buffer with a copy of another buffer. */
    Buffer& operator=(const Buffer& other)
    {
        if (this == &other)
            return *this;

        Buffer temp(other);
        swap(temp);
        return *this;
    }

    /** @brief Replaces this buffer by taking ownership from another buffer. */
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

    /** @brief Releases the owned storage and destroys constructed elements. */
    ~Buffer()
    {
        destroy(0, size_);
        deallocate(data_);
    }

    /** @brief Returns a mutable element reference without bounds checking. */
    T& operator[](std::size_t index) noexcept
    {
        return data_[index];
    }

    /** @brief Returns a const element reference without bounds checking. */
    const T& operator[](std::size_t index) const noexcept
    {
        return data_[index];
    }

    /** @brief Returns the first element. */
    T& front()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    /** @brief Returns the first element as a const reference. */
    const T& front() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    /** @brief Returns the last element. */
    T& back()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    /** @brief Returns the last element as a const reference. */
    const T& back() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    /** @brief Returns the raw data pointer. */
    [[nodiscard]] T* data() noexcept
    {
        return data_;
    }

    /** @brief Returns the raw data pointer as a const pointer. */
    [[nodiscard]] const T* data() const noexcept
    {
        return data_;
    }

    /** @brief Returns an iterator to the first element. */
    [[nodiscard]] iterator begin() noexcept
    {
        return data_;
    }

    /** @brief Returns a const iterator to the first element. */
    [[nodiscard]] const_iterator begin() const noexcept
    {
        return data_;
    }

    /** @brief Returns a const iterator to the first element. */
    [[nodiscard]] const_iterator cbegin() const noexcept
    {
        return data_;
    }

    /** @brief Returns an iterator one past the last element. */
    [[nodiscard]] iterator end() noexcept
    {
        return empty() ? data_ : data_ + size_;
    }

    /** @brief Returns a const iterator one past the last element. */
    [[nodiscard]] const_iterator end() const noexcept
    {
        return empty() ? data_ : data_ + size_;
    }

    /** @brief Returns a const iterator one past the last element. */
    [[nodiscard]] const_iterator cend() const noexcept
    {
        return empty() ? data_ : data_ + size_;
    }

    /** @brief Returns a reverse iterator to the last element. */
    [[nodiscard]] reverse_iterator rbegin() noexcept
    {
        return reverse_iterator(end());
    }

    /** @brief Returns a const reverse iterator to the last element. */
    [[nodiscard]] const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator(end());
    }

    /** @brief Returns a const reverse iterator to the last element. */
    [[nodiscard]] const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator(cend());
    }

    /** @brief Returns a reverse iterator before the first element. */
    [[nodiscard]] reverse_iterator rend() noexcept
    {
        return reverse_iterator(begin());
    }

    /** @brief Returns a const reverse iterator before the first element. */
    [[nodiscard]] const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator(begin());
    }

    /** @brief Returns a const reverse iterator before the first element. */
    [[nodiscard]] const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator(cbegin());
    }

    /** @brief Returns the number of stored elements. */
    [[nodiscard]] std::size_t size() const noexcept
    {
        return size_;
    }

    /** @brief Returns whether the buffer contains no elements. */
    [[nodiscard]] bool empty() const noexcept
    {
        return size_ == 0;
    }

    /** @brief Fills every element with the same value. */
    void fill(const T& value)
    {
        std::fill_n(data_, size_, value);
    }

    /** @brief Swaps the contents of two buffers. */
    void swap(Buffer& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

private:
    T* data_;
    std::size_t size_;

    /** @brief Allocates aligned raw storage for a number of elements. */
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

    /** @brief Releases aligned raw storage previously allocated by the buffer. */
    static void deallocate(T* ptr) noexcept
    {
        if (ptr == nullptr)
            return;

        ::operator delete(
            ptr,
            std::align_val_t{Alignment}
        );
    }

    /** @brief Fills a range by copy-constructing each value in place. */
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

    /** @brief Value-constructs a range of elements in place. */
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

    /** @brief Destroys a range of constructed elements in place. */
    void destroy(std::size_t begin, std::size_t end) noexcept
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (std::size_t index = begin; index < end; ++index) {
                std::destroy_at(data_ + index);
            }
        }
    }
}; // class Buffer

} // namespace stratax::core
