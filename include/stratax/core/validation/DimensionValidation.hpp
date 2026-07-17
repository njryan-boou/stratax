#pragma once

#include <cstddef>
#include <limits>

#include <stratax/core/Exceptions.hpp>

namespace stratax::core::validation {

/**
 * @brief Converts a signed value to `std::size_t` after rejecting negatives.
 *
 * @param value Signed value to validate.
 * @param message Error message used when `value` is negative.
 *
 * @return `value` converted to `std::size_t`.
 *
 * @throws Exceptions::DimensionError If `value` is negative.
 */
inline std::size_t nonnegative_size(std::ptrdiff_t value, const char* message)
{
    if (value < 0)
    {
        throw Exceptions::DimensionError(message);
    }

    return static_cast<std::size_t>(value);
}

/**
 * @brief Requires a rank value to match an expected rank.
 *
 * @param actual Actual rank.
 * @param expected Required rank.
 * @param message Error message used on mismatch.
 *
 * @throws Exceptions::DimensionError If the ranks differ.
 */
inline void require_rank(std::size_t actual, std::size_t expected, const char* message)
{
    if (actual != expected)
    {
        throw Exceptions::DimensionError(message);
    }
}

/**
 * @brief Requires an object with `rank()` to match an expected rank.
 *
 * @param object Object to validate.
 * @param expected Required rank.
 * @param message Error message used on mismatch.
 *
 * @return Reference to `object`.
 *
 * @throws Exceptions::DimensionError If the object rank differs.
 */
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

/**
 * @brief Multiplies two sizes while checking for overflow.
 *
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @param message Error message used on overflow.
 *
 * @return Product of `lhs` and `rhs`.
 *
 * @throws Exceptions::DimensionError If the multiplication would overflow.
 */
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

/**
 * @brief Adds two sizes while checking for overflow.
 *
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @param message Error message used on overflow.
 *
 * @return Sum of `lhs` and `rhs`.
 *
 * @throws Exceptions::DimensionError If the addition would overflow.
 */
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

}
