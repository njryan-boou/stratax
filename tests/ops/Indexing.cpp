#include <array>
#include <cassert>
#include <limits>
#include <stdexcept>

#include <stratax.hpp>

void test_offset_rank_1()
{
    stratax::core::Shape shape{5};
    stratax::core::Strides strides(shape);

    assert(offset(shape, strides, std::array<std::size_t, 1>{0}) == 0);
    assert(offset(shape, strides, std::array<std::size_t, 1>{4}) == 4);
}

void test_offset_rank_zero()
{
    stratax::core::Shape shape;
    stratax::core::Strides strides(shape);

    assert(offset(shape, strides, std::array<std::size_t, 0>{}) == 0);
}

void test_offset_rank_2()
{
    stratax::core::Shape shape{2, 3};
    stratax::core::Strides strides(shape);

    assert(offset(shape, strides, std::array<std::size_t, 2>{0, 0}) == 0);
    assert(offset(shape, strides, std::array<std::size_t, 2>{0, 2}) == 2);
    assert(offset(shape, strides, std::array<std::size_t, 2>{1, 0}) == 3);
    assert(offset(shape, strides, std::array<std::size_t, 2>{1, 2}) == 5);
}

void test_offset_rank_3()
{
    stratax::core::Shape shape{2, 3, 4};
    stratax::core::Strides strides(shape);

    assert(offset(shape, strides, std::array<std::size_t, 3>{0, 0, 0}) == 0);
    assert(offset(shape, strides, std::array<std::size_t, 3>{0, 1, 2}) == 6);
    assert(offset(shape, strides, std::array<std::size_t, 3>{1, 0, 2}) == 14);
    assert(offset(shape, strides, std::array<std::size_t, 3>{1, 2, 3}) == 23);
}

void test_offset_rejects_rank_mismatch()
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

    assert(threw);
}

void test_offset_rejects_shape_stride_rank_mismatch()
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

    assert(threw);
}

void test_offset_rejects_out_of_bounds()
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

    assert(row_threw);
    assert(col_threw);
}

void test_offset_rejects_zero_dimension_index()
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

    assert(threw);
}

void test_offset_overflow_throws()
{
    stratax::core::Shape shape{std::numeric_limits<std::size_t>::max(), 2};
    stratax::core::Strides strides(stratax::core::Shape{std::numeric_limits<std::size_t>::max(), 2});

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

    assert(threw);
}

int main()
{
    test_offset_rank_zero();
    test_offset_rank_1();
    test_offset_rank_2();
    test_offset_rank_3();
    test_offset_rejects_rank_mismatch();
    test_offset_rejects_shape_stride_rank_mismatch();
    test_offset_rejects_out_of_bounds();
    test_offset_rejects_zero_dimension_index();
    test_offset_overflow_throws();

    return 0;
}
