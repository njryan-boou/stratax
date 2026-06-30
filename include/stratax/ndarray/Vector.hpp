#pragma once

#include <cstddef>
#include <initializer_list>

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Shape.hpp>

namespace container {

template<typename T>
requires stratax::core::Numeric<T>
class Vector
{
private:
    stratax::core::Shape shape_;
    stratax::core::Buffer<T> buffer_;

public:
    // Constructors

    Vector() noexcept = default;

    explicit Vector(std::size_t size)
        : shape_{size},
          buffer_(size)
    {
    }

    Vector(std::size_t size, const T& value)
        : shape_{size},
          buffer_(size, value)
    {
    }

    Vector(std::initializer_list<T> list)
        : shape_{list.size()},
          buffer_(list)
    {
    }

    // Rule of Five

    Vector(const Vector&) = default;
    Vector(Vector&&) noexcept = default;

    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) noexcept = default;

    ~Vector() = default;


    // Element access

    T& operator[](std::size_t index);

    const T& operator[](std::size_t index) const;

    T& at(std::size_t index);

    const T& at(std::size_t index) const;

    // Capacity

    std::size_t size() const noexcept;

    bool empty() const noexcept;

    // Shape

    const stratax::core::Shape& shape() const noexcept;

    // Raw storage

    T* data() noexcept;

    const T* data() const noexcept;

    // Iterators

    auto begin() noexcept;
    auto end() noexcept;

    auto begin() const noexcept;
    auto end() const noexcept;

    // Utilities

    void fill(const T& value);

    void swap(Vector& other) noexcept;
};

}
