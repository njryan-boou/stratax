#include <gtest/gtest.h>
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

#include <stratax.h>

TEST(OpsIndexing, offset_rank_1)
{
    stratax::core::Shape shape{5};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 1>{0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 1>{4}) == 4);
}

TEST(OpsIndexing, offset_rank_zero)
{
    stratax::core::Shape shape;
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 0>{}) == 0);
}

TEST(OpsIndexing, offset_rank_2)
{
    stratax::core::Shape shape{2, 3};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 2}) == 2);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{1, 0}) == 3);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{1, 2}) == 5);
}

TEST(OpsIndexing, offset_rank_3)
{
    stratax::core::Shape shape{2, 3, 4};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 3>{0, 0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 3>{0, 1, 2}) == 6);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 3>{1, 0, 2}) == 14);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 3>{1, 2, 3}) == 23);
}

TEST(OpsIndexing, offset_rejects_rank_mismatch)
{
    stratax::core::Shape shape{2, 3, 4};
    stratax::core::Strides strides(shape);

    bool threw = false;

    try {
        offset(shape, strides, std::array<std::size_t, 2>{1, 2});
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsIndexing, offset_rejects_shape_stride_rank_mismatch)
{
    stratax::core::Shape shape{2, 3};
    stratax::core::Strides strides(stratax::core::Shape{2, 3, 4});

    bool threw = false;

    try {
        offset(shape, strides, std::array<std::size_t, 2>{1, 2});
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsIndexing, offset_rejects_out_of_bounds)
{
    stratax::core::Shape shape{2, 3};
    stratax::core::Strides strides(shape);

    bool row_threw = false;
    bool col_threw = false;

    try {
        offset(shape, strides, std::array<std::size_t, 2>{2, 0});
    }
    catch (const Exceptions::IndexError&) {
        row_threw = true;
    }

    try {
        offset(shape, strides, std::array<std::size_t, 2>{0, 3});
    }
    catch (const Exceptions::IndexError&) {
        col_threw = true;
    }

    EXPECT_TRUE(row_threw);
    EXPECT_TRUE(col_threw);
}

TEST(OpsIndexing, offset_rejects_zero_dimension_index)
{
    stratax::core::Shape shape{2, 0, 3};
    stratax::core::Strides strides(shape);

    bool threw = false;

    try {
        offset(shape, strides, std::array<std::size_t, 3>{0, 0, 0});
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsIndexing, offset_overflow_throws)
{
    stratax::core::Shape shape{std::numeric_limits<long long>::max(), 2};
    stratax::core::Strides strides(stratax::core::Shape{std::numeric_limits<long long>::max(), 2});

    bool threw = false;

    try {
        offset(shape, strides, std::array<std::size_t, 2>{
            std::numeric_limits<std::size_t>::max() - 1,
            1
        });
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

