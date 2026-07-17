#include <gtest/gtest.h>

#include <type_traits>

#include <stratax/core/Exceptions.hpp>

TEST(CoreExceptions, exception_hierarchy)
{
    static_assert(std::is_base_of_v<std::runtime_error, Exceptions::StrataxError>);
    static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::ShapeError>);
    static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::DimensionError>);
    static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::IndexError>);
    static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::TypeError>);
    static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::BroadcastError>);
    static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::ZeroDivisionError>);
}

TEST(CoreExceptions, exception_message_roundtrip)
{
    const char* message = "shape mismatch";

    try {
        throw Exceptions::ShapeError(message);
    }
    catch (const Exceptions::ShapeError& error) {
        EXPECT_STREQ(error.what(), message);
        return;
    }

    FAIL();
}
