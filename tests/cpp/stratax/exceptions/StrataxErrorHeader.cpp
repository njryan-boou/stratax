#include <stratax/exceptions/StrataxError.hpp>

#include <type_traits>

static_assert(std::is_base_of_v<std::runtime_error, Exceptions::StrataxError>);
