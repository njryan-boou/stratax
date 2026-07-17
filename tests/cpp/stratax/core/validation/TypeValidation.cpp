#include <gtest/gtest.h>

#include <string>

#include <stratax/core/containers/Vector.hpp>
#include <stratax/core/validation/TypeValidation.hpp>

namespace validation = stratax::core::validation;

TEST(CoreTypeValidation, require_type)
{
    EXPECT_NO_THROW((validation::require_type<int, int>("bad")));
    EXPECT_THROW((validation::require_type<int, float>("bad")), Exceptions::TypeError);
}

TEST(CoreTypeValidation, require_numeric_type)
{
    EXPECT_NO_THROW((validation::require_numeric_type<double>("bad")));
    EXPECT_THROW((validation::require_numeric_type<std::string>("bad")), Exceptions::TypeError);
}

TEST(CoreTypeValidation, require_same_value_type)
{
    const stratax::container::Vector<int> ints_a{1, 2};
    const stratax::container::Vector<int> ints_b{3, 4};
    const stratax::container::Vector<float> floats{1.0f, 2.0f};

    EXPECT_NO_THROW(validation::require_same_value_type(ints_a, ints_b, "bad"));
    EXPECT_THROW(validation::require_same_value_type(ints_a, floats, "bad"), Exceptions::TypeError);
}
