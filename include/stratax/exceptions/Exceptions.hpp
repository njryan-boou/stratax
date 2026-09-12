/** @file
 * @brief Stratax exception types derived from std::runtime_error.
 */
#pragma once

#include <stdexcept>

namespace Exceptions {

/** @brief Base class for Stratax errors; derives from std::runtime_error. */
class StrataxError : public std::runtime_error
{
public:
	using std::runtime_error::runtime_error;
};

/** @brief Shape incompatible with the requested operation or container. */
class ShapeError : public StrataxError { public: using StrataxError::StrataxError; };
/** @brief Invalid dimension or overflow in dimension, element-count, or stride arithmetic. */
class DimensionError : public StrataxError { public: using StrataxError::StrataxError; };
/** @brief Number of axes or supplied multidimensional indices does not match the required rank. */
class RankError : public StrataxError { public: using StrataxError::StrataxError; };
/** @brief Index or slice cannot be used for the requested access. */
class IndexError : public StrataxError { public: using StrataxError::StrataxError; };
/** @brief Unsupported type conversion or type-related operation. */
class TypeError : public StrataxError { public: using StrataxError::StrataxError; };
/** @brief Operands cannot broadcast to the requested result. */
class BroadcastError : public StrataxError { public: using StrataxError::StrataxError; };
/** @brief An explicitly checked divisor is zero. */
class ZeroDivisionError : public StrataxError { public: using StrataxError::StrataxError; };
/** @brief Reduction axis is outside the input rank. */
class AxisError : public StrataxError { public: using StrataxError::StrataxError; };
/** @brief Exception type reserved for explicitly reported numeric overflow; arithmetic does not automatically check for overflow. */
class OverflowError : public StrataxError { public: using StrataxError::StrataxError; };
/** @brief Invalid argument value, such as an invalid shift count or null nonempty view. */
class ValueError : public StrataxError { public: using StrataxError::StrataxError; };

} // namespace Exceptions
