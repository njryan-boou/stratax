#include <gtest/gtest.h>
#include <limits>

#include <stratax.h>

namespace validation = stratax::core::validation;

TEST(CoreValidation, nonnegative_size_accepts_zero_and_positive)
{
    EXPECT_TRUE(validation::nonnegative_size(0, "bad") == 0);
    EXPECT_TRUE(validation::nonnegative_size(7, "bad") == 7);
}

TEST(CoreValidation, nonnegative_size_rejects_negative)
{
    bool threw = false;

    try
    {
        validation::nonnegative_size(-1, "negative size");
    }
    catch (const Exceptions::DimensionError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreValidation, nonnegative_shape_dimension_rejects_negative)
{
    bool threw = false;

    try
    {
        validation::nonnegative_shape_dimension(-1, "negative dimension");
    }
    catch (const Exceptions::ShapeError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreValidation, nonnegative_index_rejects_negative)
{
    bool threw = false;

    try
    {
        validation::nonnegative_index(-1, "negative index");
    }
    catch (const Exceptions::IndexError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreValidation, require_rank_accepts_matching_rank)
{
    stratax::core::Shape shape{2, 3};

    validation::require_rank(2, 2, "rank mismatch");
    const auto& returned = validation::require_rank(shape, 2, "rank mismatch");

    EXPECT_TRUE(&returned == &shape);
}

TEST(CoreValidation, require_rank_rejects_mismatch)
{
    bool threw = false;

    try
    {
        validation::require_rank(1, 2, "rank mismatch");
    }
    catch (const Exceptions::DimensionError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreValidation, require_rank_object_overload_rejects_mismatch)
{
    stratax::core::Shape shape{2, 3};

    bool threw = false;

    try
    {
        validation::require_rank(shape, 3, "rank mismatch");
    }
    catch (const Exceptions::DimensionError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreValidation, require_index_accepts_valid_index)
{
    validation::require_index(0, 1, "bad index");
    validation::require_index(2, 3, "bad index");
}

TEST(CoreValidation, require_index_rejects_size_and_beyond)
{
    bool exact_size_threw = false;

    try
    {
        validation::require_index(3, 3, "bad index");
    }
    catch (const Exceptions::IndexError&)
    {
        exact_size_threw = true;
    }

    EXPECT_TRUE(exact_size_threw);

    bool beyond_size_threw = false;

    try
    {
        validation::require_index(4, 3, "bad index");
    }
    catch (const Exceptions::IndexError&)
    {
        beyond_size_threw = true;
    }

    EXPECT_TRUE(beyond_size_threw);
}

TEST(CoreValidation, require_at_most_accepts_upper_bound)
{
    validation::require_at_most(0, 3, "too large");
    validation::require_at_most(3, 3, "too large");
}

TEST(CoreValidation, require_at_most_rejects_larger_value)
{
    bool threw = false;

    try
    {
        validation::require_at_most(4, 3, "too large");
    }
    catch (const Exceptions::IndexError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreValidation, checked_multiply_accepts_safe_product)
{
    EXPECT_TRUE(validation::checked_multiply(6, 7, "overflow") == 42);
    EXPECT_TRUE(validation::checked_multiply(0, std::numeric_limits<std::size_t>::max(), "overflow") == 0);
}

TEST(CoreValidation, checked_multiply_rejects_overflow)
{
    bool threw = false;

    try
    {
        validation::checked_multiply(
            std::numeric_limits<std::size_t>::max(),
            2,
            "overflow");
    }
    catch (const Exceptions::DimensionError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreValidation, checked_add_accepts_safe_sum)
{
    EXPECT_TRUE(validation::checked_add(10, 20, "overflow") == 30);
}

TEST(CoreValidation, checked_add_rejects_overflow)
{
    bool threw = false;

    try
    {
        validation::checked_add(
            std::numeric_limits<std::size_t>::max(),
            1,
            "overflow");
    }
    catch (const Exceptions::DimensionError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreValidation, same_shape_and_require_same_shape)
{
    stratax::container::Vector<int> a{1, 2, 3};
    stratax::container::Vector<int> b{4, 5, 6};
    stratax::container::Vector<int> c{7, 8};

    EXPECT_TRUE(validation::same_shape(a, b));
    EXPECT_TRUE(!validation::same_shape(a, c));

    validation::require_same_shape(a, b, "shape mismatch");

    bool threw = false;

    try
    {
        validation::require_same_shape(a, c, "shape mismatch");
    }
    catch (const Exceptions::ShapeError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreValidation, require_equal_size)
{
    validation::require_equal_size(4, 4, "size mismatch");

    bool threw = false;

    try
    {
        validation::require_equal_size(4, 5, "size mismatch");
    }
    catch (const Exceptions::ShapeError&)
    {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

