#include <stratax/exceptions/IndexErrors.hpp>

#include <type_traits>

static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::IndexError>);
static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::AxisError>);
