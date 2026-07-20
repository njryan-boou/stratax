#include <gtest/gtest.h>
#include <concepts>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <stratax.h>

using namespace stratax::core;

static_assert(std::constructible_from<Shape, int>);
static_assert(std::constructible_from<Shape, int, int, int>);
static_assert(!std::constructible_from<Shape, bool>);
static_assert(!std::constructible_from<Shape, char>);
static_assert(!std::constructible_from<Shape, double>);

TEST(CoreShape, default_constructor)
{
    Shape shape;

    EXPECT_TRUE(shape.rank() == 0);
    EXPECT_TRUE(shape.empty());
    EXPECT_TRUE(shape.elements() == 0);
    EXPECT_TRUE(shape.begin() == shape.end());
}

TEST(CoreShape, initializer_list_constructor)
{
    Shape shape{3, 224, 224};

    EXPECT_TRUE(shape.rank() == 3);
    EXPECT_TRUE(shape.elements() == 150528);
    EXPECT_TRUE(shape(0) == 3);
    EXPECT_TRUE(shape(1) == 224);
    EXPECT_TRUE(shape(2) == 224);
}

TEST(CoreShape, zero_dimension_elements)
{
    Shape scalar_zero{0};
    Shape matrix_zero{0, 3};
    Shape later_zero{std::numeric_limits<long long>::max(), 0};

    EXPECT_TRUE(scalar_zero.rank() == 1);
    EXPECT_TRUE(scalar_zero.elements() == 0);
    EXPECT_TRUE(matrix_zero.rank() == 2);
    EXPECT_TRUE(matrix_zero.elements() == 0);
    EXPECT_TRUE(later_zero.rank() == 2);
    EXPECT_TRUE(later_zero.elements() == 0);
}

TEST(CoreShape, initializer_list_constructor_rejects_bool_and_char_like_types)
{
    // Compile-time checks above cover bool, char-like, and floating-point input.
    Shape shape{0};
    EXPECT_TRUE(shape.rank() == 1);
    EXPECT_TRUE(shape.elements() == 0);
}

TEST(CoreShape, elements_overflow_throws)
{
    Shape shape{2, std::numeric_limits<long long>::max(), 2};

    EXPECT_THROW(
        [[maybe_unused]] const std::size_t elements = shape.elements(),
        Exceptions::StrataxError);
}

TEST(CoreShape, buffer_constructors)
{
    Buffer<std::size_t> dims{2, 3, 4};
    Shape copied(dims);

    EXPECT_TRUE(copied.rank() == 3);
    EXPECT_TRUE(copied.elements() == 24);
    EXPECT_TRUE(copied(0) == 2);
    EXPECT_TRUE(copied(1) == 3);
    EXPECT_TRUE(copied(2) == 4);

    Shape moved(Buffer<std::size_t>{5, 6});

    EXPECT_TRUE(moved.rank() == 2);
    EXPECT_TRUE(moved.elements() == 30);
    EXPECT_TRUE(moved(0) == 5);
    EXPECT_TRUE(moved(1) == 6);
}

TEST(CoreShape, std_vector_constructor)
{
    std::vector<std::size_t> dims{7, 8, 9};
    Shape shape(dims);

    EXPECT_TRUE(shape.rank() == 3);
    EXPECT_TRUE(shape.elements() == 504);
    EXPECT_TRUE(shape(0) == 7);
    EXPECT_TRUE(shape(1) == 8);
    EXPECT_TRUE(shape(2) == 9);

    dims[0] = 1;
    EXPECT_TRUE(shape(0) == 7);
}

TEST(CoreShape, std_vector_constructor_empty)
{
    std::vector<std::size_t> dims;
    Shape shape(dims);

    EXPECT_TRUE(shape.rank() == 0);
    EXPECT_TRUE(shape.empty());
    EXPECT_TRUE(shape.elements() == 0);
}

TEST(CoreShape, index_operater)
{
    Shape shape{3, 224, 224};

    EXPECT_TRUE(shape(0) == 3);
    EXPECT_TRUE(shape(1) == 224);
    EXPECT_TRUE(shape(2) == 224);

    bool threw = false;

    try {
        shape(3);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreShape, negative_index_operator)
{
    Shape shape{3, 224, 224};

    EXPECT_TRUE(shape[-1] == 224);
    EXPECT_TRUE(shape[-2] == 224);
    EXPECT_TRUE(shape[-3] == 3);

    bool threw = false;

    try {
        [[maybe_unused]] auto dim = shape[-4];
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreShape, equality)
{
    Shape a{3, 224, 224};
    Shape b{3, 224, 224};
    Shape c{224, 224};
    Shape d{224, 3, 224};
    Shape empty_a;
    Shape empty_b;

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));

    EXPECT_TRUE(a != c);
    EXPECT_TRUE(!(a == c));

    EXPECT_TRUE(a != d);
    EXPECT_TRUE(!(a == d));

    EXPECT_TRUE(empty_a == empty_b);
    EXPECT_TRUE(!(empty_a != empty_b));
}

TEST(CoreShape, iteration)
{
    Shape shape{3, 224, 224};

    std::size_t expected[] = {3, 224, 224};
    std::size_t i = 0;

    for (std::size_t dim : shape) {
        EXPECT_TRUE(dim == expected[i]);
        ++i;
    }

    EXPECT_TRUE(i == 3);
}

TEST(CoreShape, output)
{
    Shape shape{3, 224, 224};
    Shape rank_one{3};
    Shape empty;

    std::ostringstream out;
    out << shape;

    EXPECT_TRUE(out.str() == "(3, 224, 224)");

    std::ostringstream rank_one_out;
    rank_one_out << rank_one;

    EXPECT_TRUE(rank_one_out.str() == "(3,)");

    std::ostringstream empty_out;
    empty_out << empty;

    EXPECT_TRUE(empty_out.str() == "()");
}

TEST(CoreShape, copy_assignment)
{
    Shape original{2, 3, 4};
    Shape copy;

    copy = original;

    EXPECT_TRUE(copy == original);
    EXPECT_TRUE(copy.rank() == 3);
    EXPECT_TRUE(copy.elements() == 24);

    copy = copy;
    EXPECT_TRUE(copy == original);
}

TEST(CoreShape, move_assignment)
{
    Shape original{5, 6};
    Shape moved;

    moved = std::move(original);

    EXPECT_TRUE(moved.rank() == 2);
    EXPECT_TRUE(moved.elements() == 30);
    EXPECT_TRUE(moved(0) == 5);
    EXPECT_TRUE(moved(1) == 6);

    Shape& same = moved;
    moved = std::move(same);
    EXPECT_TRUE(moved.rank() == 2);
    EXPECT_TRUE(moved.elements() == 30);
}

TEST(CoreShape, swap)
{
    Shape a{2, 3, 4};
    Shape b{5};

    a.swap(b);

    EXPECT_TRUE(a.rank() == 1);
    EXPECT_TRUE(a(0) == 5);
    EXPECT_TRUE(b.rank() == 3);
    EXPECT_TRUE(b(0) == 2);
    EXPECT_TRUE(b(1) == 3);
    EXPECT_TRUE(b(2) == 4);
}

// ============================================================================
// Copy Constructor & Semantics
// ============================================================================

TEST(CoreShape, copy_constructor)
{
    Shape original{2, 3, 4};
    Shape copy(original);

    EXPECT_TRUE(copy == original);
    EXPECT_TRUE(copy.rank() == 3);
    EXPECT_TRUE(copy.elements() == 24);
    EXPECT_TRUE(copy(0) == 2);
    EXPECT_TRUE(copy(1) == 3);
    EXPECT_TRUE(copy(2) == 4);

    // Verify independence
    Shape modified{10, 20, 30};
    copy = modified;

    EXPECT_TRUE(original != copy);
    EXPECT_TRUE(original(0) == 2);
    EXPECT_TRUE(copy(0) == 10);
}

TEST(CoreShape, copy_constructor_empty)
{
    Shape original;
    Shape copy(original);

    EXPECT_TRUE(copy == original);
    EXPECT_TRUE(copy.empty());
    EXPECT_TRUE(copy.rank() == 0);
}

// ============================================================================
// Allow Zero Tag Constructors
// ============================================================================

TEST(CoreShape, allow_zero_initializer_list_constructor)
{
    Shape shape(std::initializer_list<std::size_t>{0, 5, 0, 3}, Shape::allow_zero);

    EXPECT_TRUE(shape.rank() == 4);
    EXPECT_TRUE(shape(0) == 0);
    EXPECT_TRUE(shape(1) == 5);
    EXPECT_TRUE(shape(2) == 0);
    EXPECT_TRUE(shape(3) == 3);
    EXPECT_TRUE(shape.elements() == 0);
}

TEST(CoreShape, allow_zero_buffer_constructor)
{
    Buffer<std::size_t> dims{1, 0, 2};
    Shape shape(dims, Shape::allow_zero);

    EXPECT_TRUE(shape.rank() == 3);
    EXPECT_TRUE(shape(1) == 0);
    EXPECT_TRUE(shape.elements() == 0);
}

// ============================================================================
// Const Method Coverage
// ============================================================================

TEST(CoreShape, const_operator_parenthesis)
{
    const Shape shape{3, 224, 224};

    EXPECT_TRUE(shape(0) == 3);
    EXPECT_TRUE(shape(1) == 224);
    EXPECT_TRUE(shape(2) == 224);
}

TEST(CoreShape, const_operator_bracket_negative_index)
{
    const Shape shape{3, 224, 224};

    EXPECT_TRUE(shape[-1] == 224);
    EXPECT_TRUE(shape[-2] == 224);
    EXPECT_TRUE(shape[-3] == 3);
}

TEST(CoreShape, const_empty)
{
    const Shape empty;
    const Shape non_empty{1, 2, 3};

    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(!non_empty.empty());
}

TEST(CoreShape, const_rank)
{
    const Shape shape{5, 6, 7, 8};

    EXPECT_TRUE(shape.rank() == 4);
}

TEST(CoreShape, const_elements)
{
    const Shape shape{2, 3, 4};

    EXPECT_TRUE(shape.elements() == 24);
}

TEST(CoreShape, const_equality)
{
    const Shape a{1, 2, 3};
    const Shape b{1, 2, 3};
    const Shape c{1, 2};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a != c);
}

TEST(CoreShape, const_iteration)
{
    const Shape shape{10, 20, 30};

    std::size_t expected[] = {10, 20, 30};
    std::size_t i = 0;

    for (std::size_t dim : shape) {
        EXPECT_TRUE(dim == expected[i]);
        ++i;
    }

    EXPECT_TRUE(i == 3);
}

TEST(CoreShape, const_begin_end)
{
    const Shape shape{7, 8, 9};

    const std::size_t* begin_ptr = shape.begin();
    const std::size_t* end_ptr = shape.end();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 7);
}

// ============================================================================
// Mutable Iterator Coverage
// ============================================================================

TEST(CoreShape, mutable_begin_end)
{
    Shape shape{1, 2, 3};

    std::size_t* begin_ptr = shape.begin();
    std::size_t* end_ptr = shape.end();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 1);
}

TEST(CoreShape, mutable_iteration)
{
    Shape shape{5, 6, 7};

    std::size_t i = 0;
    for (std::size_t dim : shape) {
        ++i;
    }

    EXPECT_TRUE(i == 3);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(CoreShape, single_dimension_shape)
{
    Shape shape{42};

    EXPECT_TRUE(shape.rank() == 1);
    EXPECT_TRUE(shape.elements() == 42);
    EXPECT_TRUE(shape(0) == 42);
}

TEST(CoreShape, large_dimension_count)
{
    // Create shape with many dimensions
    std::vector<std::size_t> dims(100, 1);
    Shape shape(dims);

    EXPECT_TRUE(shape.rank() == 100);
    EXPECT_TRUE(shape.elements() == 1);
    EXPECT_TRUE(shape(50) == 1);
    EXPECT_TRUE(shape[-1] == 1);
}

TEST(CoreShape, very_large_dimension_value)
{
    // Test with single large dimension
    Shape shape{std::numeric_limits<std::uint32_t>::max()};

    EXPECT_TRUE(shape.rank() == 1);
    EXPECT_TRUE(shape(0) == std::numeric_limits<std::uint32_t>::max());
}

TEST(CoreShape, multiple_zero_dimensions)
{
    Shape shape{0, 0, 0};

    EXPECT_TRUE(shape.rank() == 3);
    EXPECT_TRUE(shape.elements() == 0);
    EXPECT_TRUE(shape(0) == 0);
    EXPECT_TRUE(shape(1) == 0);
    EXPECT_TRUE(shape(2) == 0);
}

TEST(CoreShape, zero_in_middle)
{
    Shape shape{5, 0, 3};

    EXPECT_TRUE(shape.rank() == 3);
    EXPECT_TRUE(shape.elements() == 0);
    EXPECT_TRUE(shape(1) == 0);
}

// ============================================================================
// Index Access Edge Cases
// ============================================================================

TEST(CoreShape, operator_parenthesis_out_of_bounds)
{
    Shape shape{1, 2, 3};

    bool threw = false;
    try {
        shape(10);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreShape, operator_bracket_negative_out_of_bounds)
{
    Shape shape{1, 2, 3};

    bool threw = false;
    try {
        shape[-10];
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreShape, operator_parenthesis_on_empty_throws)
{
    Shape shape;

    bool threw = false;
    try {
        shape(0);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

// ============================================================================
// Swap Edge Cases
// ============================================================================

TEST(CoreShape, swap_with_empty)
{
    Shape filled{1, 2, 3};
    Shape empty;

    filled.swap(empty);

    EXPECT_TRUE(filled.empty());
    EXPECT_TRUE(filled.rank() == 0);
    EXPECT_TRUE(empty.rank() == 3);
    EXPECT_TRUE(empty(0) == 1);
}

TEST(CoreShape, swap_empty_with_empty)
{
    Shape a;
    Shape b;

    a.swap(b);

    EXPECT_TRUE(a.empty());
    EXPECT_TRUE(b.empty());
}

// ============================================================================
// Inequality Operator
// ============================================================================

TEST(CoreShape, inequality_different_ranks)
{
    Shape a{1, 2, 3};
    Shape b{1, 2};

    EXPECT_TRUE(a != b);
    EXPECT_TRUE(!(a == b));
}

TEST(CoreShape, inequality_different_values_same_rank)
{
    Shape a{1, 2, 3};
    Shape b{1, 2, 4};

    EXPECT_TRUE(a != b);
    EXPECT_TRUE(!(a == b));
}

// ============================================================================
// Stream Output Edge Cases
// ============================================================================

TEST(CoreShape, output_single_dimension)
{
    Shape shape{42};
    std::ostringstream out;
    out << shape;

    EXPECT_TRUE(out.str() == "(42,)");
}

TEST(CoreShape, output_empty_shape)
{
    Shape shape;
    std::ostringstream out;
    out << shape;

    EXPECT_TRUE(out.str() == "()");
}

TEST(CoreShape, output_with_zeros)
{
    Shape shape{0, 5, 0};
    std::ostringstream out;
    out << shape;

    EXPECT_TRUE(out.str() == "(0, 5, 0)");
}

