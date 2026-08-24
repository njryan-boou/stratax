#include <stratax/exceptions/TypeErrors.hpp>

#include <type_traits>

static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::TypeError>);
static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::ValueError>);
