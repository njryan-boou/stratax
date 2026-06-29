#pragma once
#include "Buffer.hpp"

#include <stdexcept>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <iostream>

namespace core {

class Shape
{
private:
    Buffer<std::size_t> dims_;

public:
    Shape() noexcept : dims_()
    {
    }

    //initialized list
    Shape(std::initializer_list<std::size_t> list) : dims_(list)
    {
    }

    ~Shape() = default;

    std::size_t size() const
    {
        int prod = 1;
        for (int i = 0; i < dims_.size(); i++)
        {
            prod *=  dims_[i];
        }
        return prod;
    }

    std::size_t rank() const
    {
        return dims_.size();
    }

    const std::size_t& operator[](std::size_t index) const
    {   
        if (index >= rank()) 
        {
            throw std::out_of_range("Shape dimension index out of bounds");
        }
        return dims_[index];
    }

    bool empty() const noexcept
    {
        return dims_.empty();
    }

    bool operator==(const Shape& other) const noexcept
    {
        if (rank() != other.rank())
        {
            return false;
        }
        for (std::size_t i = 0; i < rank(); ++i)
        {
            if (dims_[i] != other.dims_[i])
            {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const Shape& other) const noexcept
    {
        return !(*this == other);
    }

    auto begin() noexcept
    {
        return dims_.begin();
    }

    auto end() noexcept
    {
        return dims_.end();
    }

    auto begin() const noexcept
    {
        return dims_.begin();
    }

    auto end() const noexcept
    {
        return dims_.end();
    }

};

std::ostream& operator<<(std::ostream& os, const Shape& shape)
    {
        os << "(";
        bool first = true;
        for (std::size_t dim : shape)
        {
            if (!first)
                os << ", ";

            os << dim;
            first = false;
        }

        os << ')';
        return os;
    }

}
