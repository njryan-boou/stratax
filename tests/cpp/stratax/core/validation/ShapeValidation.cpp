#include <gtest/gtest.h>
#include <stratax/core/validation/ShapeValidation.hpp>
#include <stratax/core/Shape.hpp>

TEST(ShapeValidation, RequiresMatchingShapes)
{
    const stratax::core::Shape actual{2, 3};
    const stratax::core::Shape expected{3, 2};

    EXPECT_NO_THROW(stratax::validation::require_same_shape(actual, actual));

    try
    {
        stratax::validation::require_same_shape(actual, expected);
        FAIL() << "Expected Exceptions::ShapeError";
    }
    catch (const Exceptions::ShapeError& error)
    {
        EXPECT_EQ(error.actual_shape(), (Exceptions::ShapeError::shape_type{2, 3}));
        EXPECT_EQ(error.expected_shape(), (Exceptions::ShapeError::shape_type{3, 2}));
    }
}
