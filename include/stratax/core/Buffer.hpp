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

#include <stratax/core/Config.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::core {

template<typename T, std::size_t Alignment = config::default_alignment>
class Buffer
{
    static_assert(Alignment >= alignof(T));
    static_assert((Alignment & (Alignment - 1)) == 0);

public:
    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    Buffer() noexcept
        : data_(nullptr), size_(0)
    {
    }

    explicit Buffer(std::size_t size)
        : data_(allocate(size)), size_(size)
    {
        construct_default();
    }

    Buffer(std::size_t size, const T& value)
        : data_(allocate(size)), size_(size)
    {
        construct_fill(value);
    }

    Buffer(std::initializer_list<T> list)
        : data_(allocate(list.size())), size_(list.size())
    {
        std::size_t i = 0;

        try {
            for (const T& value : list) {
                std::construct_at(data_ + i, value);
                ++i;
            }
        } catch (...) {
            destroy(i);
            deallocate(data_);
            throw;
        }
    }

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
            destroy(i);
            deallocate(data_);
            throw;
        }
    }

    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Buffer& operator=(const Buffer& other)
    {
        if (this == &other) {
            return *this;
        }

        Buffer temp(other);
        swap(temp);
        return *this;
    }

    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }

        destroy(size_);
        deallocate(data_);

        data_ = other.data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;

        return *this;
    }

    ~Buffer()
    {
        destroy(size_);
        deallocate(data_);
    }

    T& operator[](std::size_t index) noexcept
    {
        return data_[index];
    }

    const T& operator[](std::size_t index) const noexcept
    {
        return data_[index];
    }

    T& front()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    const T& front() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    T& back()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    const T& back() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    [[nodiscard]] T* data() noexcept
    {
        return data_;
    }

    [[nodiscard]] const T* data() const noexcept
    {
        return data_;
    }

    iterator begin() noexcept
    {
        return data_;
    }

    const_iterator begin() const noexcept
    {
        return data_;
    }

    const_iterator cbegin() const noexcept
    {
        return data_;
    }

    iterator end() noexcept
    {
        return empty() ? data_ : data_ + size_;
    }

    const_iterator end() const noexcept
    {
        return empty() ? data_ : data_ + size_;
    }

    const_iterator cend() const noexcept
    {
        return empty() ? data_ : data_ + size_;
    }

    reverse_iterator rbegin() noexcept
    {
        return reverse_iterator(end());
    }

    const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator(end());
    }

    const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator(cend());
    }

    reverse_iterator rend() noexcept
    {
        return reverse_iterator(begin());
    }

    const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator(begin());
    }

    const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator(cbegin());
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return size_;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return size_ == 0;
    }

    void fill(const T& value)
    {
        std::fill_n(data_, size_, value);
    }

    void swap(Buffer& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

private:
    T* data_;
    std::size_t size_;

    static T* allocate(std::size_t space)
    {
        if (space == 0) {
            return nullptr;
        }

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

    static void deallocate(T* ptr) noexcept
    {
        if (ptr == nullptr) {
            return;
        }

        ::operator delete(
            ptr,
            std::align_val_t{Alignment}
        );
    }

    void construct_fill(const T& value)
    {
        try {
            std::uninitialized_fill_n(data_, size_, value);
        } catch (...) {
            deallocate(data_);
            data_ = nullptr;
            size_ = 0;
            throw;
        }
    }

    void construct_default()
    {
        try {
            std::uninitialized_value_construct_n(data_, size_);
        } catch (...) {
            deallocate(data_);
            data_ = nullptr;
            size_ = 0;
            throw;
        }
    }

    void destroy(std::size_t count) noexcept
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (std::size_t index = 0; index < count; ++index) {
                std::destroy_at(data_ + index);
            }
        }
    }
};

}
