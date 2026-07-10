#include <cassert>
#include <limits>

#include "stratax.hpp"

namespace validation = stratax::core::validation;

void test_nonnegative_size_accepts_zero_and_positive()
{
    assert(validation::nonnegative_size(0, "bad") == 0);
    assert(validation::nonnegative_size(7, "bad") == 7);
}

void test_nonnegative_size_rejects_negative()
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

    assert(threw);
}

void test_nonnegative_shape_dimension_rejects_negative()
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

    assert(threw);
}

void test_nonnegative_index_rejects_negative()
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

    assert(threw);
}

void test_require_rank_accepts_matching_rank()
{
    stratax::core::Shape shape{2, 3};

    validation::require_rank(2, 2, "rank mismatch");
    const auto& returned = validation::require_rank(shape, 2, "rank mismatch");

    assert(&returned == &shape);
}

void test_require_rank_rejects_mismatch()
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

    assert(threw);
}

void test_require_rank_object_overload_rejects_mismatch()
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

    assert(threw);
}

void test_require_index_accepts_valid_index()
{
    validation::require_index(0, 1, "bad index");
    validation::require_index(2, 3, "bad index");
}

void test_require_index_rejects_size_and_beyond()
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

    assert(exact_size_threw);

    bool beyond_size_threw = false;

    try
    {
        validation::require_index(4, 3, "bad index");
    }
    catch (const Exceptions::IndexError&)
    {
        beyond_size_threw = true;
    }

    assert(beyond_size_threw);
}

void test_require_at_most_accepts_upper_bound()
{
    validation::require_at_most(0, 3, "too large");
    validation::require_at_most(3, 3, "too large");
}

void test_require_at_most_rejects_larger_value()
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

    assert(threw);
}

void test_checked_multiply_accepts_safe_product()
{
    assert(validation::checked_multiply(6, 7, "overflow") == 42);
    assert(validation::checked_multiply(0, std::numeric_limits<std::size_t>::max(), "overflow") == 0);
}

void test_checked_multiply_rejects_overflow()
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

    assert(threw);
}

void test_checked_add_accepts_safe_sum()
{
    assert(validation::checked_add(10, 20, "overflow") == 30);
}

void test_checked_add_rejects_overflow()
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

    assert(threw);
}

void test_same_shape_and_require_same_shape()
{
    stratax::container::Vector<int> a{1, 2, 3};
    stratax::container::Vector<int> b{4, 5, 6};
    stratax::container::Vector<int> c{7, 8};

    assert(validation::same_shape(a, b));
    assert(!validation::same_shape(a, c));

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

    assert(threw);
}

void test_require_equal_size()
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

    assert(threw);
}

int main()
{
    test_nonnegative_size_accepts_zero_and_positive();
    test_nonnegative_size_rejects_negative();
    test_nonnegative_shape_dimension_rejects_negative();
    test_nonnegative_index_rejects_negative();
    test_require_rank_accepts_matching_rank();
    test_require_rank_rejects_mismatch();
    test_require_rank_object_overload_rejects_mismatch();
    test_require_index_accepts_valid_index();
    test_require_index_rejects_size_and_beyond();
    test_require_at_most_accepts_upper_bound();
    test_require_at_most_rejects_larger_value();
    test_checked_multiply_accepts_safe_product();
    test_checked_multiply_rejects_overflow();
    test_checked_add_accepts_safe_sum();
    test_checked_add_rejects_overflow();
    test_same_shape_and_require_same_shape();
    test_require_equal_size();

    return 0;
}
