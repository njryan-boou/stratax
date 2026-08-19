#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::core::validation {

inline std::size_t nonnegative_size(std::ptrdiff_t value, const char* message)
{
    if (value < 0)
    {
        throw Exceptions::DimensionError(message);
    }

    return static_cast<std::size_t>(value);
}

inline void require_rank(std::size_t actual, std::size_t expected, const char* message)
{
    if (actual != expected)
    {
        throw Exceptions::DimensionError(message);
    }
}

template<typename Ranked>
requires requires(const Ranked& object)
{
    object.rank();
}
const Ranked& require_rank(const Ranked& object, std::size_t expected, const char* message)
{
    require_rank(object.rank(), expected, message);
    return object;
}

inline std::size_t checked_multiply(
    std::size_t lhs,
    std::size_t rhs,
    const char* message)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    {
        throw Exceptions::DimensionError(message);
    }

    return lhs * rhs;
}

inline std::size_t checked_add(
    std::size_t lhs,
    std::size_t rhs,
    const char* message)
{
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    {
        throw Exceptions::DimensionError(message);
    }

    return lhs + rhs;
}

inline std::size_t nonnegative_index(std::ptrdiff_t value, const char* message)
{
    if (value < 0)
    {
        throw Exceptions::IndexError(message);
    }

    return static_cast<std::size_t>(value);
}

inline void require_index(std::size_t index, std::size_t size, const char* message)
{
    if (index >= size)
    {
        throw Exceptions::IndexError(message);
    }
}

inline void require_at_most(std::size_t value, std::size_t upper, const char* message)
{
    if (value > upper)
    {
        throw Exceptions::IndexError(message);
    }
}

inline std::size_t nonnegative_shape_dimension(std::ptrdiff_t value, const char* message)
{
    if (value < 0)
    {
        throw Exceptions::ShapeError(message);
    }

    return static_cast<std::size_t>(value);
}

inline std::size_t positive_shape_dimension(std::ptrdiff_t value, const char* message)
{
    if (value <= 0)
    {
        throw Exceptions::ShapeError(message);
    }

    return static_cast<std::size_t>(value);
}

inline void require_positive_shape_dimension(std::size_t value, const char* message)
{
    if (value == 0)
    {
        throw Exceptions::ShapeError(message);
    }
}

template<typename Lhs, typename Rhs>
[[nodiscard]] bool same_shape(const Lhs& lhs, const Rhs& rhs)
{
    return lhs.size() == rhs.size() && lhs.shape() == rhs.shape();
}

template<typename Lhs, typename Rhs>
void require_same_shape(const Lhs& lhs, const Rhs& rhs, const char* message)
{
    if (!same_shape(lhs, rhs))
    {
        throw Exceptions::ShapeError(message);
    }
}

inline void require_equal_size(std::size_t lhs, std::size_t rhs, const char* message)
{
    if (lhs != rhs)
    {
        throw Exceptions::ShapeError(message);
    }
}

template<typename Actual, typename Expected>
void require_type(const char* message)
{
    if constexpr (!std::same_as<std::remove_cvref_t<Actual>, std::remove_cvref_t<Expected>>)
    {
        throw Exceptions::TypeError(message);
    }
}

template<typename T>
void require_numeric_type(const char* message)
{
    if constexpr (!Numeric<T>)
    {
        throw Exceptions::TypeError(message);
    }
}

template<typename Lhs, typename Rhs>
requires requires
{
    typename Lhs::value_type;
    typename Rhs::value_type;
}
void require_same_value_type(const Lhs& lhs, const Rhs& rhs, const char* message)
{
    (void)lhs;
    (void)rhs;
    require_type<typename Lhs::value_type, typename Rhs::value_type>(message);
}

} // namespace stratax::core::validation
