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

TEST(OpsIndexing, offset_rank_4)
{
    stratax::core::Shape shape{2, 3, 4, 5};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 4>{0, 0, 0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 4>{1, 2, 3, 4}) == 119);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 4>{1, 1, 1, 1}) == 86);
}

TEST(OpsIndexing, offset_rank_5)
{
    stratax::core::Shape shape{2, 2, 2, 2, 2};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 5>{0, 0, 0, 0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 5>{1, 1, 1, 1, 1}) == 31);
}

TEST(OpsIndexing, offset_single_element_1d)
{
    stratax::core::Shape shape{1};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 1>{0}) == 0);
}

TEST(OpsIndexing, offset_single_element_2d)
{
    stratax::core::Shape shape{1, 1};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 0}) == 0);
}

TEST(OpsIndexing, offset_single_row_matrix)
{
    stratax::core::Shape shape{1, 5};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 4}) == 4);
}

TEST(OpsIndexing, offset_single_column_matrix)
{
    stratax::core::Shape shape{5, 1};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{4, 0}) == 4);
}

TEST(OpsIndexing, offset_large_matrix)
{
    stratax::core::Shape shape{50, 50};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{25, 25}) == 1275);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{49, 49}) == 2499);
}

TEST(OpsIndexing, offset_rectangular_matrix_3x2)
{
    stratax::core::Shape shape{3, 2};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{1, 1}) == 3);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{2, 1}) == 5);
}

TEST(OpsIndexing, offset_rectangular_matrix_2x3)
{
    stratax::core::Shape shape{2, 3};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{1, 2}) == 5);
}

TEST(OpsIndexing, offset_boundary_indices_2d)
{
    stratax::core::Shape shape{10, 10};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{9, 9}) == 99);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{9, 0}) == 90);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 2>{0, 9}) == 9);
}

TEST(OpsIndexing, offset_boundary_indices_3d)
{
    stratax::core::Shape shape{5, 5, 5};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 3>{4, 4, 4}) == 124);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 3>{4, 0, 0}) == 100);
}

TEST(OpsIndexing, offset_rejects_out_of_bounds_first_dim)
{
    stratax::core::Shape shape{5, 5};
    stratax::core::Strides strides(shape);

    bool threw = false;

    try {
        offset(shape, strides, std::array<std::size_t, 2>{5, 0});
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsIndexing, offset_rejects_out_of_bounds_last_dim)
{
    stratax::core::Shape shape{5, 5};
    stratax::core::Strides strides(shape);

    bool threw = false;

    try {
        offset(shape, strides, std::array<std::size_t, 2>{0, 5});
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsIndexing, offset_rejects_out_of_bounds_middle_dim)
{
    stratax::core::Shape shape{5, 5, 5};
    stratax::core::Strides strides(shape);

    bool threw = false;

    try {
        offset(shape, strides, std::array<std::size_t, 3>{2, 5, 2});
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsIndexing, offset_rejects_rank_mismatch_fewer_indices)
{
    stratax::core::Shape shape{5, 5, 5};
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

TEST(OpsIndexing, offset_rejects_rank_mismatch_more_indices)
{
    stratax::core::Shape shape{5, 5};
    stratax::core::Strides strides(shape);

    bool threw = false;

    try {
        offset(shape, strides, std::array<std::size_t, 3>{1, 2, 3});
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(OpsIndexing, offset_zero_dimension_in_middle)
{
    stratax::core::Shape shape{5, 0, 5};
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

TEST(OpsIndexing, offset_large_3d_tensor)
{
    stratax::core::Shape shape{10, 20, 30};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 3>{0, 0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 3>{9, 19, 29}) == 5999);
}

TEST(OpsIndexing, offset_rank_6)
{
    stratax::core::Shape shape{2, 2, 2, 2, 2, 2};
    stratax::core::Strides strides(shape);

    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 6>{0, 0, 0, 0, 0, 0}) == 0);
    EXPECT_TRUE(offset(shape, strides, std::array<std::size_t, 6>{1, 1, 1, 1, 1, 1}) == 63);
}

TEST(OpsIndexing, offset_consistency_row_major_2d)
{
    stratax::core::Shape shape{3, 4};
    stratax::core::Strides strides(shape);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            std::size_t flat_offset = offset(shape, strides, std::array<std::size_t, 2>{i, j});
            EXPECT_TRUE(flat_offset == i * 4 + j);
        }
    }
}

TEST(OpsIndexing, offset_consistency_row_major_3d)
{
    stratax::core::Shape shape{2, 3, 4};
    stratax::core::Strides strides(shape);

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                std::size_t flat_offset = offset(shape, strides, std::array<std::size_t, 3>{i, j, k});
                EXPECT_TRUE(flat_offset == i * 12 + j * 4 + k);
            }
        }
    }
}

