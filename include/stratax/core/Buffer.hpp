#pragma once

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <utility>
#include <new>
#include <type_traits>

#include "Config.hpp"

namespace stratax::core {

template<typename T, std::size_t Alignment = 64>
class Buffer {
public:
    struct uninitialized_t {};
    static constexpr uninitialized_t uninitialized{};

    Buffer() noexcept
        : data_(nullptr), size_(0) {}

    explicit Buffer(std::size_t size)
        : data_(allocate(size)), size_(size)
    {
        construct_default(0, size_);
    }

    Buffer(std::size_t size, uninitialized_t)
        : data_(allocate(size)), size_(size)
    {
        static_assert(
            std::is_trivially_destructible_v<T>,
            "uninitialized Buffer storage is only safe for trivially destructible types"
        );
    }

    Buffer(std::size_t size, const T& value)
        : data_(allocate(size)), size_(size)
    {
        construct_fill(0, size_, value);
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
            destroy_range(0, i);
            deallocate(data_);
            throw;
        }
    }

    Buffer(const Buffer& other)
        : data_(allocate(other.size_)), size_(other.size_)
    {
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(data_, other.data_, sizeof(T) * size_);
            return;
        }

        std::size_t i = 0;

        try {
            for (; i < size_; ++i) {
                std::construct_at(data_ + i, other.data_[i]);
            }
        } catch (...) {
            destroy_range(0, i);
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
        if (this == &other)
            return *this;

        Buffer temp(other);
        swap(temp);
        return *this;
    }

    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this == &other)
            return *this;

        destroy_range(0, size_);
        deallocate(data_);

        data_ = other.data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;

        return *this;
    }

    ~Buffer()
    {
        destroy_range(0, size_);
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

    T& front() noexcept
    {
        return data_[0];
    }

    const T& front() const noexcept
    {
        return data_[0];
    }

    T& back() noexcept
    {
        return data_[size_ - 1];
    }

    const T& back() const noexcept
    {
        return data_[size_ - 1];
    }

    T* data() noexcept
    {
        return data_;
    }

    const T* data() const noexcept
    {
        return data_;
    }

    T* begin() noexcept
    {
        return data_;
    }

    const T* begin() const noexcept
    {
        return data_;
    }

    const T* cbegin() const noexcept
    {
        return data_;
    }

    T* end() noexcept
    {
        return data_ + size_;
    }

    const T* end() const noexcept
    {
        return data_ + size_;
    }

    const T* cend() const noexcept
    {
        return data_ + size_;
    }

    std::reverse_iterator<T*> rbegin() noexcept
    {
        return std::reverse_iterator<T*>(end());
    }

    std::reverse_iterator<const T*> rbegin() const noexcept
    {
        return std::reverse_iterator<const T*>(end());
    }

    std::reverse_iterator<const T*> crbegin() const noexcept
    {
        return std::reverse_iterator<const T*>(cend());
    }

    std::reverse_iterator<T*> rend() noexcept
    {
        return std::reverse_iterator<T*>(begin());
    }

    std::reverse_iterator<const T*> rend() const noexcept
    {
        return std::reverse_iterator<const T*>(begin());
    }

    std::reverse_iterator<const T*> crend() const noexcept
    {
        return std::reverse_iterator<const T*>(cbegin());
    }

    std::size_t size() const noexcept
    {
        return size_;
    }

    bool empty() const noexcept
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

private:
    static T* allocate(std::size_t count)
    {
        if (count == 0)
            return nullptr;

        return static_cast<T*>(
            ::operator new(
                sizeof(T) * count,
                std::align_val_t{Alignment}
            )
        );
    }

    static void deallocate(T* ptr) noexcept
    {
        ::operator delete(
            ptr,
            std::align_val_t{Alignment}
        );
    }

    void construct_default(std::size_t first, std::size_t last)
    {
        try {
            std::uninitialized_default_construct_n(data_ + first, last - first);
        } catch (...) {
            deallocate(data_);
            data_ = nullptr;
            size_ = 0;
            throw;
        }
    }

    void construct_fill(std::size_t first, std::size_t last, const T& value)
    {
        try {
            std::uninitialized_fill_n(data_ + first, last - first, value);
        } catch (...) {
            deallocate(data_);
            data_ = nullptr;
            size_ = 0;
            throw;
        }
    }

    void destroy_range(std::size_t first, std::size_t last) noexcept
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (std::size_t i = first; i < last; ++i) {
                std::destroy_at(data_ + i);
            }
        }
    }
};

}

