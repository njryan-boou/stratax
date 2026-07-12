#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

#include <stratax/core/Concepts.hpp>
#include <stratax/core/Exceptions.hpp>

/**
 * @file Validation.hpp
 * @brief Shared runtime validation helpers.
 */

namespace stratax::core::validation {

/**
 * @brief Requires a type to exactly match an expected type after cv/ref removal.
 *
 * @tparam Actual Type being validated.
 * @tparam Expected Required type.
 * @param message Error message used on mismatch.
 *
 * @throws Exceptions::TypeError If the normalized types differ.
 */
template<typename Actual, typename Expected>
void require_type(const char* message)
{
    if constexpr (!std::same_as<std::remove_cvref_t<Actual>, std::remove_cvref_t<Expected>>)
    {
        throw Exceptions::TypeError(message);
    }
}

/**
 * @brief Requires a type to satisfy Stratax's numeric type rules.
 *
 * @tparam T Type being validated.
 * @param message Error message used when the type is unsupported.
 *
 * @throws Exceptions::TypeError If `T` is not a supported numeric type.
 */
template<typename T>
void require_numeric_type(const char* message)
{
    if constexpr (!Numeric<T>)
    {
        throw Exceptions::TypeError(message);
    }
}

/**
 * @brief Requires two array-like objects to have the same value type.
 *
 * @param lhs Left object.
 * @param rhs Right object.
 * @param message Error message used on mismatch.
 *
 * @throws Exceptions::TypeError If the objects' normalized `value_type`s differ.
 */
template<typename Lhs, typename Rhs>
requires requires
{
    typename Lhs::value_type;
    typename Rhs::value_type;
}
void require_same_value_type(const Lhs&, const Rhs&, const char* message)
{
    require_type<typename Lhs::value_type, typename Rhs::value_type>(message);
}

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
inline std::size_t nonnegative_size(long long value, const char* message)
{
    if (value < 0)
    {
        throw Exceptions::DimensionError(message);
    }

    return static_cast<std::size_t>(value);
}

/**
 * @brief Converts a signed shape dimension to `std::size_t` after rejecting negatives.
 *
 * @param value Signed dimension to validate.
 * @param message Error message used when `value` is negative.
 *
 * @return `value` converted to `std::size_t`.
 *
 * @throws Exceptions::ShapeError If `value` is negative.
 */
inline std::size_t nonnegative_shape_dimension(long long value, const char* message)
{
    if (value < 0)
    {
        throw Exceptions::ShapeError(message);
    }

    return static_cast<std::size_t>(value);
}

/**
 * @brief Converts a signed shape dimension to `std::size_t` after requiring it to be positive.
 *
 * @param value Signed dimension to validate.
 * @param message Error message used when `value <= 0`.
 *
 * @return `value` converted to `std::size_t`.
 *
 * @throws Exceptions::ShapeError If `value <= 0`.
 */
inline std::size_t positive_shape_dimension(long long value, const char* message)
{
    if (value <= 0)
    {
        throw Exceptions::ShapeError(message);
    }

    return static_cast<std::size_t>(value);
}

/**
 * @brief Requires an unsigned shape dimension to be positive.
 *
 * @param value Dimension to validate.
 * @param message Error message used when `value == 0`.
 *
 * @throws Exceptions::ShapeError If `value == 0`.
 */
inline void require_positive_shape_dimension(std::size_t value, const char* message)
{
    if (value == 0)
    {
        throw Exceptions::ShapeError(message);
    }
}

/**
 * @brief Converts a signed index to `std::size_t` after rejecting negatives.
 *
 * @param value Signed index to validate.
 * @param message Error message used when `value` is negative.
 *
 * @return `value` converted to `std::size_t`.
 *
 * @throws Exceptions::IndexError If `value` is negative.
 */
inline std::size_t nonnegative_index(long long value, const char* message)
{
    if (value < 0)
    {
        throw Exceptions::IndexError(message);
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
 * @brief Requires an index to be less than a size.
 *
 * @param index Index to validate.
 * @param size Exclusive upper bound.
 * @param message Error message used when out of bounds.
 *
 * @throws Exceptions::IndexError If `index >= size`.
 */
inline void require_index(std::size_t index, std::size_t size, const char* message)
{
    if (index >= size)
    {
        throw Exceptions::IndexError(message);
    }
}

/**
 * @brief Requires a value to be less than or equal to an upper bound.
 *
 * This is useful for half-open range stops, where `stop == size` is valid.
 *
 * @param value Value to validate.
 * @param upper Inclusive upper bound.
 * @param message Error message used when `value > upper`.
 *
 * @throws Exceptions::IndexError If `value > upper`.
 */
inline void require_at_most(std::size_t value, std::size_t upper, const char* message)
{
    if (value > upper)
    {
        throw Exceptions::IndexError(message);
    }
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

/**
 * @brief Returns whether two array-like objects have identical size and shape.
 *
 * @param lhs Left object.
 * @param rhs Right object.
 *
 * @return `true` if size and shape match.
 */
template<typename Lhs, typename Rhs>
[[nodiscard]] bool same_shape(const Lhs& lhs, const Rhs& rhs)
{
    return lhs.size() == rhs.size() && lhs.shape() == rhs.shape();
}

/**
 * @brief Requires two array-like objects to have identical size and shape.
 *
 * @param lhs Left object.
 * @param rhs Right object.
 * @param message Error message used on mismatch.
 *
 * @throws Exceptions::ShapeError If size or shape differ.
 */
template<typename Lhs, typename Rhs>
void require_same_shape(const Lhs& lhs, const Rhs& rhs, const char* message)
{
    if (!same_shape(lhs, rhs))
    {
        throw Exceptions::ShapeError(message);
    }
}

/**
 * @brief Requires two sizes to be equal.
 *
 * @param lhs Left size.
 * @param rhs Right size.
 * @param message Error message used on mismatch.
 *
 * @throws Exceptions::ShapeError If the sizes differ.
 */
inline void require_equal_size(std::size_t lhs, std::size_t rhs, const char* message)
{
    if (lhs != rhs)
    {
        throw Exceptions::ShapeError(message);
    }
}

}
