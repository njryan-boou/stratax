#include <gtest/gtest.h>

#include <stratax/core/containers/Vector.hpp>
#include <stratax/core/validation/ShapeValidation.hpp>

namespace validation = stratax::core::validation;

TEST(CoreShapeValidation, dimension_checks)
{
    EXPECT_TRUE(validation::nonnegative_shape_dimension(0, "bad") == 0);
    EXPECT_TRUE(validation::positive_shape_dimension(1, "bad") == 1);

    EXPECT_THROW(validation::nonnegative_shape_dimension(-1, "bad"), Exceptions::ShapeError);
    EXPECT_THROW(validation::positive_shape_dimension(0, "bad"), Exceptions::ShapeError);
    EXPECT_THROW(validation::positive_shape_dimension(-1, "bad"), Exceptions::ShapeError);

    EXPECT_NO_THROW(validation::require_positive_shape_dimension(1, "bad"));
    EXPECT_THROW(validation::require_positive_shape_dimension(0, "bad"), Exceptions::ShapeError);
}

TEST(CoreShapeValidation, shape_comparisons)
{
    stratax::container::Vector<int> a{1, 2, 3};
    stratax::container::Vector<int> b{4, 5, 6};
    stratax::container::Vector<int> c{7, 8};

    EXPECT_TRUE(validation::same_shape(a, b));
    EXPECT_TRUE(!validation::same_shape(a, c));

    EXPECT_NO_THROW(validation::require_same_shape(a, b, "shape mismatch"));
    EXPECT_THROW(validation::require_same_shape(a, c, "shape mismatch"), Exceptions::ShapeError);

    EXPECT_NO_THROW(validation::require_equal_size(3, 3, "size mismatch"));
    EXPECT_THROW(validation::require_equal_size(3, 2, "size mismatch"), Exceptions::ShapeError);
}
