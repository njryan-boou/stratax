#pragma once

#include "Config.hpp"
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::core {

/**
 * @brief Fixed-size owner of aligned, contiguous element storage.
 *
 * Buffer allocates storage with the requested alignment and manages the
 * lifetime of every element in that storage. Its size cannot change after
 * construction, although its elements remain mutable. Elements occupy one
 * contiguous allocation, so pointers and random-access iterators can be used
 * with standard-library algorithms.
 *
 * Copy operations create an independent allocation and copy the elements.
 * Move operations transfer the allocation in constant time and leave the
 * source empty. Destruction releases every element followed by the aligned
 * allocation, giving the class exclusive RAII ownership of its storage.
 *
 * @tparam T Element type stored in the buffer.
 * @tparam Alignment Allocation alignment in bytes. It must be a power of two
 *                   and at least `alignof(T)`.
 *
 * @par Example
 * @code
 * stratax::core::Buffer<float, 64> samples{1.0F, 2.0F, 3.0F};
 *
 * samples[1] = 5.0F;
 * std::fill(samples.begin(), samples.end(), 4.0F);
 *
 * stratax::core::Buffer<float, 64> copy = samples; // Deep copy.
 * copy.front() = 9.0F;                             // samples is unchanged.
 * @endcode
 *
 * @note The class does not provide resizing or capacity management. Create or
 *       assign another Buffer when a different element count is required.
 */
template<typename T, std::size_t Alignment = config::default_alignment>
class Buffer
{

    static_assert(Alignment >= alignof(T));
    static_assert((Alignment & (Alignment - 1)) == 0);

public:
    /** @brief Type of each stored element. */
    using value_type = T;
    /** @brief Unsigned type used for element counts and indices. */
    using size_type = std::size_t;
    /** @brief Signed type used for distances between iterators. */
    using difference_type = std::ptrdiff_t;
    /** @brief Mutable element reference type. */
    using reference = T&;
    /** @brief Read-only element reference type. */
    using const_reference = const T&;
    /** @brief Mutable pointer to an element. */
    using pointer = T*;
    /** @brief Read-only pointer to an element. */
    using const_pointer = const T*;
    /** @brief Mutable contiguous random-access iterator type. */
    using iterator = pointer;
    /** @brief Read-only contiguous random-access iterator type. */
    using const_iterator = const_pointer;
    /** @brief Mutable iterator that traverses elements in reverse order. */
    using reverse_iterator = std::reverse_iterator<iterator>;
    /** @brief Read-only iterator that traverses elements in reverse order. */
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /**
     * @brief Returns the alignment, in bytes, used for buffer allocations.
     *
     * This is a compile-time property of the Buffer specialization.
     *
     * @return Requested alignment of the underlying allocation, in bytes.
     * @complexity O(1).
     */
    [[nodiscard]] static constexpr size_type alignment() noexcept
    {
        return Alignment;
    }

    /**
     * @brief Returns the largest element count that cannot overflow in bytes.
     *
     * This is an arithmetic limit rather than a promise that an allocation of
     * that size will succeed.
     *
     * @return `std::numeric_limits<size_type>::max() / sizeof(value_type)`.
     * @complexity O(1).
     */
    [[nodiscard]] static constexpr size_type max_size() noexcept
    {
        return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }

    /**
     * @brief Constructs an empty buffer that owns no allocation.
     *
     * `size()` is zero, `empty()` is true, and `data()` is `nullptr`.
     *
     * @complexity O(1).
     */
    Buffer() noexcept: data_(nullptr), size_(0) {}

    /**
     * @brief Constructs a buffer with value-initialized elements.
     *
     * Each element is initialized from `value_type{}`.
     * If element construction fails, already-constructed elements and the
     * allocation are cleaned up.
     *
     * @param size Number of elements to construct.
     * @throws std::bad_array_new_length If @p size exceeds max_size().
     * @throws std::bad_alloc If storage allocation fails.
     * @throws Any exception thrown while constructing an element.
     * @complexity O(size).
     */
    explicit Buffer(size_type size)
        : data_(allocate(size)), size_(size)
    {
        construct_fill(value_type{});
    }

    /**
     * @brief Constructs a buffer with all elements initialized to @p value.
     *
     * If an element copy fails, already-constructed elements and the
     * allocation are cleaned up.
     *
     * @param size Number of elements to construct.
     * @param value Value copied into each element.
     * @throws std::bad_array_new_length If @p size exceeds max_size().
     * @throws std::bad_alloc If storage allocation fails.
     * @throws Any exception thrown by `value_type`'s copy constructor.
     * @complexity O(size).
     */
    Buffer(size_type size, const_reference value)
        : data_(allocate(size)), size_(size)
    {
        construct_fill(value);
    }

    /**
     * @brief Constructs a buffer from an initializer list.
     *
     * Elements are copy-constructed in list order.
     * If an element copy fails, already-constructed elements and the
     * allocation are cleaned up.
     *
     * @param list Source elements.
     * @throws std::bad_array_new_length If list size exceeds max_size().
     * @throws std::bad_alloc If storage allocation fails.
     * @throws Any exception thrown by `value_type`'s copy constructor.
     * @complexity O(list.size()).
     */
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

    /**
     * @brief Copy-constructs a buffer with independent storage.
     *
     * If copying an element fails, all elements already constructed in the
     * new allocation are destroyed and the exception is propagated.
     *
     * @param other Buffer to copy from.
     * @throws std::bad_array_new_length If `other.size()` exceeds max_size().
     * @throws std::bad_alloc If storage allocation fails.
     * @throws Any exception thrown by `value_type`'s copy constructor.
     * @complexity O(other.size()).
     */
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

    /**
     * @brief Move-constructs a buffer by transferring ownership.
     *
     * After construction, @p other is empty and owns no allocation.
     * Existing pointers and iterators into @p other remain valid and now refer
     * to elements owned by the newly constructed buffer. Element move
     * constructors are not invoked.
     *
     * @param other Buffer to move from.
     * @complexity O(1).
     */
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief Copy-assigns from another buffer.
     *
     * This operation provides the strong exception guarantee: if allocation
     * or element copying fails, this buffer is unchanged. On success, previous
     * pointers, references, and iterators into this buffer are invalidated.
     *
     * @param other Buffer to copy from.
     * @return Reference to this buffer.
     * @throws std::bad_array_new_length If `other.size()` exceeds max_size().
     * @throws std::bad_alloc If storage allocation fails.
     * @throws Any exception thrown by `value_type`'s copy constructor.
     * @complexity O(size() + other.size()).
     */
    Buffer& operator=(const Buffer& other)
    {
        if (this == &other) {
            return *this;
        }

        Buffer temp(other);
        swap(temp);
        return *this;
    }

    /**
     * @brief Move-assigns from another buffer.
     *
     * Existing elements are destroyed before ownership is transferred.
     * After assignment, @p other is empty and owns no allocation.
     * Pointers and iterators into @p other remain valid and now refer to
     * elements owned by this buffer; those into this buffer's former storage
     * are invalidated.
     *
     * @param other Buffer to move from.
     * @return Reference to this buffer.
     * @complexity O(size()) to destroy the previous elements; ownership
     *             transfer is O(1).
     */
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

    /**
     * @brief Destroys the buffer and releases owned storage.
     *
     * Each live element is destroyed before the allocation is released.
     *
     * @complexity O(size()).
     */
    ~Buffer()
    {
        if (size_ != 0) {
            std::destroy_n(data_, size_);
        }

        deallocate(data_);
    }

    /**
     * @brief Returns an unchecked reference to an element.
     *
     * @param index Element position.
     * @return Reference to the element at @p index.
     * @pre `index < size()`; otherwise behavior is undefined.
     * @complexity O(1).
     */
    reference operator[](size_type index) noexcept {return data_[index];}

    /**
     * @brief Returns an unchecked const reference to an element.
     *
     * @param index Element position.
     * @return Const reference to the element at @p index.
     * @pre `index < size()`; otherwise behavior is undefined.
     * @complexity O(1).
     */
    const_reference operator[](size_type index) const noexcept {return data_[index];}

    /**
     * @brief Returns a reference to the first element.
     *
     * @return Reference to the first element.
     * @throws Exceptions::IndexError If the buffer is empty.
     * @complexity O(1).
     */
    reference front()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    /**
     * @brief Returns a const reference to the first element.
     *
     * @return Const reference to the first element.
     * @throws Exceptions::IndexError If the buffer is empty.
     * @complexity O(1).
     */
    const_reference front() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer front cannot be accessed when the buffer is empty.");
        }

        return data_[0];
    }

    /**
     * @brief Returns a reference to the last element.
     *
     * @return Reference to the last element.
     * @throws Exceptions::IndexError If the buffer is empty.
     * @complexity O(1).
     */
    reference back()
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    /**
     * @brief Returns a const reference to the last element.
     *
     * @return Const reference to the last element.
     * @throws Exceptions::IndexError If the buffer is empty.
     * @complexity O(1).
     */
    const_reference back() const
    {
        if (empty()) {
            throw Exceptions::IndexError("Buffer back cannot be accessed when the buffer is empty.");
        }

        return data_[size_ - 1];
    }

    /**
     * @brief Returns a pointer to the underlying contiguous storage.
     *
     * @return Pointer to storage, or nullptr when empty.
     *
     * Elements occupy the range `[data(), data() + size())` when non-empty.
     * The pointer remains valid until this buffer is assigned to, moved from,
     * swapped, or destroyed.
     *
     * @complexity O(1).
     */
    [[nodiscard]] pointer data() noexcept {return data_;}

    /**
     * @brief Returns a const pointer to the underlying contiguous storage.
     *
     * @return Const pointer to storage, or nullptr when empty.
     *
     * Elements occupy the range `[data(), data() + size())` when non-empty.
     * The pointer remains valid until this buffer is assigned to, moved from,
     * swapped, or destroyed.
     *
     * @complexity O(1).
     */
    [[nodiscard]] const_pointer data() const noexcept {return data_;}

    /**
     * @brief Returns a mutable iterator to the first element.
     * @return `data()`, which equals end() when the buffer is empty.
     * @complexity O(1).
     */
    iterator begin() noexcept {return data_;}
    /**
     * @brief Returns a const iterator to the first element.
     * @return `data()`, which equals end() when the buffer is empty.
     * @complexity O(1).
     */
    const_iterator begin() const noexcept {return data_;}
    /**
     * @brief Returns a const iterator to the first element.
     * @return `data()`, which equals cend() when the buffer is empty.
     * @complexity O(1).
     */
    const_iterator cbegin() const noexcept {return data_;}
    /**
     * @brief Returns a mutable iterator one past the final element.
     * @return Past-the-end iterator for the mutable element range.
     * @complexity O(1).
     */
    iterator end() noexcept {return empty() ? data_ : data_ + size_;}
    /**
     * @brief Returns a const iterator one past the final element.
     * @return Past-the-end iterator for the const element range.
     * @complexity O(1).
     */
    const_iterator end() const noexcept {return empty() ? data_ : data_ + size_;}
    /**
     * @brief Returns a const iterator one past the final element.
     * @return Past-the-end iterator for the const element range.
     * @complexity O(1).
     */
    const_iterator cend() const noexcept {return empty() ? data_ : data_ + size_;}
    /**
     * @brief Returns a mutable reverse iterator to the final element.
     * @return Reverse iterator constructed from end().
     * @complexity O(1).
     */
    reverse_iterator rbegin() noexcept {return reverse_iterator(end());}
    /**
     * @brief Returns a const reverse iterator to the final element.
     * @return Const reverse iterator constructed from end().
     * @complexity O(1).
     */
    const_reverse_iterator rbegin() const noexcept {return const_reverse_iterator(end());}
    /**
     * @brief Returns a const reverse iterator to the final element.
     * @return Const reverse iterator constructed from cend().
     * @complexity O(1).
     */
    const_reverse_iterator crbegin() const noexcept {return const_reverse_iterator(cend());}
    /**
     * @brief Returns the past-the-end iterator for mutable reverse traversal.
     * @return Reverse iterator constructed from begin().
     * @complexity O(1).
     */
    reverse_iterator rend() noexcept {return reverse_iterator(begin());}
    /**
     * @brief Returns the past-the-end iterator for const reverse traversal.
     * @return Const reverse iterator constructed from begin().
     * @complexity O(1).
     */
    const_reverse_iterator rend() const noexcept {return const_reverse_iterator(begin());}
    /**
     * @brief Returns the past-the-end iterator for const reverse traversal.
     * @return Const reverse iterator constructed from cbegin().
     * @complexity O(1).
     */
    const_reverse_iterator crend() const noexcept {return const_reverse_iterator(cbegin());}

    /**
     * @brief Returns the number of stored elements.
     *
     * @return Element count.
     *
     * The count is fixed between construction and assignment.
     *
     * @complexity O(1).
     */
    [[nodiscard]] size_type size() const noexcept {return size_;}

    /**
     * @brief Returns whether the buffer has no elements.
     *
     * @return `true` if size() is zero; otherwise `false`.
     * @complexity O(1).
     */
    [[nodiscard]] bool empty() const noexcept {return size_ == 0;}

    /**
     * @brief Assigns a value to every element in the buffer.
     *
     * Assignment proceeds from the first element to the last. If an assignment
     * throws, earlier elements retain their new values while the remaining
     * elements keep their previous values.
     *
     * @param value Value copied into each element.
     * @throws Any exception thrown by `value_type`'s copy-assignment operator.
     * @complexity O(size()).
     */
    void fill(const_reference value)
    {
        std::fill_n(data_, size_, value);
    }

    /**
     * @brief Swaps storage and size with another buffer.
     *
     * @param other Buffer to swap with.
     *
     * This operation does not move, copy, or swap individual elements.
     * Iterators and pointers continue to refer to the same elements, now owned
     * by the other buffer. References are likewise unaffected.
     *
     * @complexity O(1).
     */
    void swap(Buffer& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

private:
    pointer data_;
    size_type size_;

    /**
     * @brief Allocates aligned, uninitialized storage for @p space elements.
     *
     * @param space Number of elements to allocate storage for.
     * @return Pointer to uninitialized storage, or nullptr when @p space is zero.
     *
     * @throws std::bad_array_new_length If @p space exceeds max_size().
     * @throws std::bad_alloc If allocation fails.
     * @complexity O(1), excluding allocator-internal work.
     */
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

    /**
     * @brief Releases storage previously returned by allocate().
     *
     * @param ptr Storage pointer to release; `nullptr` is ignored.
     * @complexity O(1), excluding allocator-internal work.
     */
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

    /**
     * @brief Constructs each element by copying @p value.
     *
     * @param value Source value for element construction.
     *
     * If construction throws, the owned allocation is released and the
     * exception is propagated. `data_` is not reset because construction of
     * the enclosing Buffer will not complete.
     *
     * @complexity O(size()).
     */
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
