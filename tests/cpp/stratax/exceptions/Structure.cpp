#include <gtest/gtest.h>

#include <stratax/exceptions/ArithmeticErrors.hpp>
#include <stratax/exceptions/ErrorCode.hpp>
#include <stratax/exceptions/IndexErrors.hpp>
#include <stratax/exceptions/LayoutErrors.hpp>
#include <stratax/exceptions/StrataxError.hpp>
#include <stratax/exceptions/TypeErrors.hpp>

#include <string>
#include <type_traits>

static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::ShapeError>);
static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::IndexError>);
static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::TypeError>);
static_assert(std::is_base_of_v<Exceptions::StrataxError, Exceptions::OverflowError>);

TEST(ExceptionStructure, FamilyHeadersAreIndependentlyUsable)
{
    EXPECT_EQ(
        Exceptions::error_code_name(Exceptions::ErrorCode::Broadcast),
        "broadcast");
}

TEST(StrataxErrorDiagnostic, CombinesCommonMetadata)
{
    const Exceptions::AxisError error(-2, 1);
    const std::string diagnostic = error.diagnostic();

    EXPECT_NE(diagnostic.find("axis: Axis -2 is out of range for rank 1."),
        std::string::npos);
    EXPECT_NE(diagnostic.find("Structure.cpp"), std::string::npos);
}
