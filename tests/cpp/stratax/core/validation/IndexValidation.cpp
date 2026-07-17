#include <gtest/gtest.h>

#include <stratax/core/validation/IndexValidation.hpp>

namespace validation = stratax::core::validation;

TEST(CoreIndexValidation, nonnegative_index)
{
    EXPECT_TRUE(validation::nonnegative_index(0, "bad") == 0);
    EXPECT_TRUE(validation::nonnegative_index(4, "bad") == 4);
    EXPECT_THROW(validation::nonnegative_index(-1, "bad"), Exceptions::IndexError);
}

TEST(CoreIndexValidation, normalize_index)
{
    EXPECT_TRUE(validation::normalize_index(0, 3, "bad") == 0);
    EXPECT_TRUE(validation::normalize_index(-1, 3, "bad") == 2);
    EXPECT_TRUE(validation::normalize_index(-3, 3, "bad") == 0);

    EXPECT_THROW(validation::normalize_index(3, 3, "bad"), Exceptions::IndexError);
    EXPECT_THROW(validation::normalize_index(-4, 3, "bad"), Exceptions::IndexError);
}

TEST(CoreIndexValidation, range_requirements)
{
    EXPECT_NO_THROW(validation::require_index(0, 1, "bad"));
    EXPECT_THROW(validation::require_index(1, 1, "bad"), Exceptions::IndexError);

    EXPECT_NO_THROW(validation::require_at_most(3, 3, "bad"));
    EXPECT_THROW(validation::require_at_most(4, 3, "bad"), Exceptions::IndexError);
}
