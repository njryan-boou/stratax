#pragma once

#include <type_traits>

#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>

namespace stratax::core::validation {

/** @brief Requires a type to exactly match an expected type after cv/ref removal. */
template<typename Actual, typename Expected>
void require_type(const char* message)
{
    if constexpr (!std::same_as<std::remove_cvref_t<Actual>, std::remove_cvref_t<Expected>>)
    {
        throw Exceptions::TypeError(message);
    }
}

/** @brief Requires a type to satisfy Stratax's numeric type rules. */
template<typename T>
void require_numeric_type(const char* message)
{
    if constexpr (!Numeric<T>)
    {
        throw Exceptions::TypeError(message);
    }
}

/** @brief Requires two array-like objects to have the same value type. */
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

}
