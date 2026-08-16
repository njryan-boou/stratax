#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
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
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    [[nodiscard]] static constexpr size_type alignment() noexcept
    {
        return Alignment;
    }

    [[nodiscard]] static constexpr size_type max_size() noexcept
    {
        return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }

    Buffer() noexcept: data_(nullptr), size_(0) {}

    explicit Buffer(size_type size)
        : data_(allocate(size)), size_(size)
    {
        construct_fill(value_type{});
    }

    Buffer(size_type size, const_reference value)
        : data_(allocate(size)), size_(size)
    {
        construct_fill(value);
    }

    Buffer(std::initializer_list<value_type> list)
        : data_(allocate(list.size())), size_(list.size())
    {
        size_type i = 0;

        try {
            for (const_reference value : list) {
                std::construct_at(data_ + i, value);
                ++i;
            }
        } catch (...) {
            std::destroy_n(data_, i);
            deallocate(data_);
            throw;
        }
    }

    Buffer(const Buffer& other)
        : data_(allocate(other.size_)),
          size_(other.size_)
    {
        try {
            std::uninitialized_copy_n(
                other.data_,
                size_,
                data_
            );
        }
        catch (...) {
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

        if (size_ != 0) {
            std::destroy_n(data_, size_);
        }

        deallocate(data_);

        data_ = other.data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;

        return *this;
    }

    ~Buffer()
    {
        if (size_ != 0) {
            std::destroy_n(data_, size_);
        }

        deallocate(data_);
    }

    reference operator[](size_type index) noexcept {return data_[index];}
    const_reference operator[](size_type index) const noexcept {return data_[index];}

    reference front()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    const_reference front() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    reference back()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    const_reference back() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    [[nodiscard]] pointer data() noexcept {return data_;}
    [[nodiscard]] const_pointer data() const noexcept {return data_;}

    iterator begin() noexcept {return data_;}
    const_iterator begin() const noexcept {return data_;}
    const_iterator cbegin() const noexcept {return data_;}
    iterator end() noexcept {return empty() ? data_ : data_ + size_;}
    const_iterator end() const noexcept {return empty() ? data_ : data_ + size_;}
    const_iterator cend() const noexcept {return empty() ? data_ : data_ + size_;}
    reverse_iterator rbegin() noexcept {return reverse_iterator(end());}
    const_reverse_iterator rbegin() const noexcept {return const_reverse_iterator(end());}
    const_reverse_iterator crbegin() const noexcept {return const_reverse_iterator(cend());}
    reverse_iterator rend() noexcept {return reverse_iterator(begin());}
    const_reverse_iterator rend() const noexcept {return const_reverse_iterator(begin());}
    const_reverse_iterator crend() const noexcept {return const_reverse_iterator(cbegin());}

    [[nodiscard]] size_type size() const noexcept {return size_;}

    [[nodiscard]] bool empty() const noexcept {return size_ == 0;}

    void fill(const_reference value)
    {
        std::fill_n(data_, size_, value);
    }

    void swap(Buffer& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

private:
    pointer data_;
    size_type size_;

    static pointer allocate(size_type space)
    {
        if (space == 0) {
            return nullptr;
        }

        if (space > max_size()) {
            throw std::bad_array_new_length();
        }

        return static_cast<pointer>(
            ::operator new(
                sizeof(value_type) * space,
                std::align_val_t{Alignment}
            )
        );
    }

    static void deallocate(pointer ptr) noexcept
    {
        if (ptr == nullptr) {
            return;
        }

        ::operator delete(
            ptr,
            std::align_val_t{Alignment}
        );
    }

    void construct_fill(const_reference value)
    {
        try {
            std::uninitialized_fill_n(data_, size_, value);
        } catch (...) {
            deallocate(data_);
            throw;
        }
    }

};

}
