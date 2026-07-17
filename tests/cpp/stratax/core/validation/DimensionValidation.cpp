#include <gtest/gtest.h>

#include <limits>

#include <stratax/core/containers/Shape.hpp>
#include <stratax/core/validation/DimensionValidation.hpp>

namespace validation = stratax::core::validation;

TEST(CoreDimensionValidation, nonnegative_size)
{
    EXPECT_TRUE(validation::nonnegative_size(0, "bad") == 0);
    EXPECT_TRUE(validation::nonnegative_size(3, "bad") == 3);
    EXPECT_THROW(validation::nonnegative_size(-1, "bad"), Exceptions::DimensionError);
}

TEST(CoreDimensionValidation, require_rank)
{
    const stratax::core::Shape shape{2, 3};

    EXPECT_NO_THROW(validation::require_rank(2, 2, "rank mismatch"));
    EXPECT_NO_THROW(validation::require_rank(shape, 2, "rank mismatch"));
    EXPECT_THROW(validation::require_rank(1, 2, "rank mismatch"), Exceptions::DimensionError);
    EXPECT_THROW(validation::require_rank(shape, 3, "rank mismatch"), Exceptions::DimensionError);
}

TEST(CoreDimensionValidation, checked_arithmetic)
{
    EXPECT_TRUE(validation::checked_multiply(6, 7, "overflow") == 42);
    EXPECT_TRUE(validation::checked_add(10, 20, "overflow") == 30);

    EXPECT_THROW(
        validation::checked_multiply(std::numeric_limits<std::size_t>::max(), 2, "overflow"),
        Exceptions::DimensionError);
    EXPECT_THROW(
        validation::checked_add(std::numeric_limits<std::size_t>::max(), 1, "overflow"),
        Exceptions::DimensionError);
}
