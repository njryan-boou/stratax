#pragma once

#include <array>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/indexing/Indexing.hpp>

namespace stratax::core {

template<Numeric T>
class ArrayBase
{
private:

    void validate_empty() const
    {
        if (empty())
        {
            throw Exceptions::IndexError("Array is empty.");
        }
    }
public:
    

    using value_type = T;
    using iterator = typename Buffer<T>::iterator;
    using const_iterator = typename Buffer<T>::const_iterator;
    using reverse_iterator = typename Buffer<T>::reverse_iterator;
    using const_reverse_iterator = typename Buffer<T>::const_reverse_iterator;

    std::size_t size() const noexcept
    {
        return buffer_.size();
    }

    bool empty() const noexcept
    {
        return buffer_.empty();
    }

    std::size_t rank() const noexcept
	{
		return shape_.rank();
	}

    const Shape& shape() const noexcept
    {
        return shape_;
    }

    const Strides& strides() const noexcept
    {
        return strides_;
    }

	T* data() noexcept
	{
		return buffer_.data();
	}

	const T* data() const noexcept
	{
		return buffer_.data();
	}

    T& front()
    {
        validate_empty();

        return buffer_.front();
    }

    const T& front() const
    {
        validate_empty();

        return buffer_.front();
    }

    T& back()
    {
        validate_empty();

        return buffer_.back();
    }

    const T& back() const
    {
        validate_empty();

        return buffer_.back();
    }

    T& operator[](std::size_t index) noexcept
    {
        return buffer_[index];
    }

    const T& operator[](std::size_t index) const noexcept
    {
        return buffer_[index];
    }

    iterator begin() noexcept
    {
        return buffer_.begin();
    }

    const_iterator begin() const noexcept
    {
        return buffer_.begin();
    }

    const_iterator cbegin() const noexcept
    {
        return buffer_.cbegin();
    }

    iterator end() noexcept
    {
        return buffer_.end();
    }

    const_iterator end() const noexcept
    {
        return buffer_.end();
    }

    const_iterator cend() const noexcept
    {
        return buffer_.cend();
    }

    reverse_iterator rbegin() noexcept
    {
        return buffer_.rbegin();
    }

    const_reverse_iterator rbegin() const noexcept
    {
        return buffer_.rbegin();
    }

    const_reverse_iterator crbegin() const noexcept
    {
        return buffer_.crbegin();
    }

    reverse_iterator rend() noexcept
    {
        return buffer_.rend();
    }

    const_reverse_iterator rend() const noexcept
    {
        return buffer_.rend();
    }

    const_reverse_iterator crend() const noexcept
    {
        return buffer_.crend();
    }

    void fill(const T& value)
    {
        buffer_.fill(value);
    }

    void swap(ArrayBase& other) noexcept
    {
        using std::swap;

        swap(shape_, other.shape_);
        swap(strides_, other.strides_);
        swap(buffer_, other.buffer_);
    }

protected:

    ArrayBase() = default;

    explicit ArrayBase(const Shape& shape)
        : buffer_(shape.elements()),
          shape_(shape),
          strides_(shape)
    {}

    ArrayBase(const Shape& shape, const T& value)
        : buffer_(shape.elements(), value),
          shape_(shape),
          strides_(shape)
    {}

    void set_shape_and_strides(const Shape& shape)
    {
        shape_ = shape;
        strides_ = Strides(shape_);
    }

    void allocate_from_shape()
    {
        buffer_ = Buffer<T>(shape_.elements());
    }

    void allocate_from_shape(const T& value)
    {
        buffer_ = Buffer<T>(shape_.elements(), value);
    }

    void allocate_with_size(std::size_t size)
    {
        buffer_ = Buffer<T>(size);
    }

    void allocate_with_size(std::size_t size, const T& value)
    {
        buffer_ = Buffer<T>(size, value);
    }

    template<std::size_t N>
    std::array<std::size_t, N> normalize_multi_indices(
        const std::array<std::ptrdiff_t, N>& raw_indices,
        const char* rank_mismatch_message = "Multi-index rank must match array rank.") const
    {
        if (N != rank())
        {
            throw Exceptions::IndexError(rank_mismatch_message);
        }

        std::array<std::size_t, N> indices{};
        for (std::size_t i = 0; i < N; ++i)
        {
            indices[i] = normalize_axis_index(raw_indices[i], shape_[i]);
        }

        return indices;
    }

    template<std::size_t N>
    std::size_t normalized_flat_offset(
        const std::array<std::ptrdiff_t, N>& raw_indices,
        const char* rank_mismatch_message = "Multi-index rank must match array rank.",
        const char* component_oob_message = nullptr) const
    {
        try
        {
            const auto indices = normalize_multi_indices(raw_indices, rank_mismatch_message);
            return flat_offset(indices);
        }
        catch (const Exceptions::DimensionError&)
        {
            throw Exceptions::IndexError(rank_mismatch_message);
        }
        catch (const Exceptions::IndexError&)
        {
            if (component_oob_message != nullptr)
            {
                throw Exceptions::IndexError(component_oob_message);
            }

            throw;
        }
    }

    template<typename IndexContainer>
    std::size_t flat_offset(const IndexContainer& indices) const
    {
        return stratax::indexing::offset(shape_, strides_, indices);
    }

    std::size_t normalize_flat_index(std::ptrdiff_t index) const
    {
        return stratax::indexing::normalize_index(index, size());
    }

    std::size_t normalize_axis_index(std::ptrdiff_t index, std::size_t extent) const
    {
        return stratax::indexing::normalize_index(index, extent);
    }

    T& checked_flat_ref(std::ptrdiff_t index)
    {
        return buffer_[normalize_flat_index(index)];
    }

    const T& checked_flat_ref(std::ptrdiff_t index) const
    {
        return buffer_[normalize_flat_index(index)];
    }

    Buffer<T> buffer_;
    Shape shape_;
    Strides strides_;
};

}
