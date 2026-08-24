#include <stratax/exceptions/ArithmeticErrors.hpp>

#include <type_traits>

static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::ZeroDivisionError>);
static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::OverflowError>);
