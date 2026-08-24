#include <stratax/exceptions/LayoutErrors.hpp>

#include <type_traits>

static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::ShapeError>);
static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::DimensionError>);
static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::BroadcastError>);
