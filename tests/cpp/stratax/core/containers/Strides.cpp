#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#include <stratax.h>

using namespace stratax::core;

TEST(CoreStrides, default_constructor)
{
    Strides strides;

    EXPECT_TRUE(strides.size() == 0);
    EXPECT_TRUE(strides.rank() == 0);
    EXPECT_TRUE(strides.empty());
    EXPECT_TRUE(strides.data() == nullptr);
    EXPECT_TRUE(strides.begin() == strides.end());

    bool front_threw = false;
    try {
        strides.front();
    }
    catch (const Exceptions::IndexError&) {
        front_threw = true;
    }

    EXPECT_TRUE(front_threw);

    bool back_threw = false;
    try {
        strides.back();
    }
    catch (const Exceptions::IndexError&) {
        back_threw = true;
    }

    EXPECT_TRUE(back_threw);
}

TEST(CoreStrides, shape_constructor)
{
    Shape shape{2, 3, 4};
    Strides strides(shape);

    EXPECT_TRUE(strides.size() == 3);
    EXPECT_TRUE(strides.rank() == 3);
    EXPECT_TRUE(!strides.empty());

    EXPECT_TRUE(strides(0) == 12);
    EXPECT_TRUE(strides(1) == 4);
    EXPECT_TRUE(strides(2) == 1);

    EXPECT_TRUE(strides.front() == 12);
    EXPECT_TRUE(strides.back() == 1);
}

TEST(CoreStrides, vector_shape_constructor)
{
    Shape shape{5};
    Strides strides(shape);

    EXPECT_TRUE(strides.size() == 1);
    EXPECT_TRUE(strides.rank() == 1);
    EXPECT_TRUE(strides(0) == 1);
}

TEST(CoreStrides, empty_shape_constructor)
{
    Shape shape;
    Strides strides(shape);

    EXPECT_TRUE(strides.size() == 0);
    EXPECT_TRUE(strides.rank() == 0);
    EXPECT_TRUE(strides.empty());
}

TEST(CoreStrides, shape_with_one_dimensions)
{
    Strides strides(Shape{1, 3, 1});

    EXPECT_TRUE(strides.rank() == 3);
    EXPECT_TRUE(strides(0) == 3);
    EXPECT_TRUE(strides(1) == 1);
    EXPECT_TRUE(strides(2) == 1);
}

TEST(CoreStrides, shape_with_zero_dimension)
{
    Strides strides(Shape{2, 0, 4});

    EXPECT_TRUE(strides.rank() == 3);
    EXPECT_TRUE(strides(0) == 0);
    EXPECT_TRUE(strides(1) == 4);
    EXPECT_TRUE(strides(2) == 1);
}

TEST(CoreStrides, large_shape_constructs)
{
    EXPECT_NO_THROW({
        Strides strides(Shape{2, std::numeric_limits<long long>::max(), 2});
        EXPECT_TRUE(strides.rank() == 3);
    });
}

TEST(CoreStrides, at)
{
    Shape shape{2, 3, 4};
    Strides strides(shape);

    EXPECT_TRUE(strides.at(0) == 12);
    EXPECT_TRUE(strides.at(1) == 4);
    EXPECT_TRUE(strides.at(2) == 1);

    bool threw = false;

    try {
        strides.at(3);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreStrides, iteration)
{
    Shape shape{2, 3, 4};
    Strides strides(shape);

    const std::size_t expected[] = {12, 4, 1};
    std::size_t i = 0;

    for (std::size_t stride : strides) {
        EXPECT_TRUE(stride == expected[i]);
        ++i;
    }

    EXPECT_TRUE(i == 3);
}

TEST(CoreStrides, const_iteration)
{
    const Strides strides(Shape{2, 3, 4});

    const std::size_t expected[] = {12, 4, 1};
    std::size_t i = 0;

    for (auto it = strides.cbegin(); it != strides.cend(); ++it) {
        EXPECT_TRUE(*it == expected[i]);
        ++i;
    }

    EXPECT_TRUE(i == 3);
}

TEST(CoreStrides, reverse_iteration)
{
    Strides strides(Shape{2, 3, 4});

    const std::size_t expected[] = {1, 4, 12};
    std::size_t i = 0;

    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
        EXPECT_TRUE(*it == expected[i]);
        ++i;
    }

    EXPECT_TRUE(i == 3);
}

TEST(CoreStrides, stream_output)
{
    std::ostringstream empty;
    empty << Strides{};
    EXPECT_TRUE(empty.str() == "()");

    std::ostringstream vector;
    vector << Strides(Shape{5});
    EXPECT_TRUE(vector.str() == "(1,)");

    std::ostringstream tensor;
    tensor << Strides(Shape{2, 3, 4});
    EXPECT_TRUE(tensor.str() == "(12, 4, 1)");
}

TEST(CoreStrides, equality)
{
    Strides a(Shape{2, 3, 4});
    Strides b(Shape{2, 3, 4});
    Strides c(Shape{3, 4});
    Strides empty_a;
    Strides empty_b;

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));

    EXPECT_TRUE(a != c);
    EXPECT_TRUE(!(a == c));

    EXPECT_TRUE(empty_a == empty_b);
    EXPECT_TRUE(!(empty_a != empty_b));
}

TEST(CoreStrides, copy_constructor)
{
    Strides original(Shape{2, 3, 4});
    Strides copy(original);

    EXPECT_TRUE(copy == original);
    EXPECT_TRUE(copy(0) == 12);
    EXPECT_TRUE(copy(1) == 4);
    EXPECT_TRUE(copy(2) == 1);
}

TEST(CoreStrides, copy_assignment)
{
    Strides original(Shape{2, 3, 4});
    Strides copy;

    copy = original;

    EXPECT_TRUE(copy == original);
    EXPECT_TRUE(copy.rank() == 3);

    copy = copy;
    EXPECT_TRUE(copy == original);
    EXPECT_TRUE(copy.rank() == 3);
}

TEST(CoreStrides, move_constructor)
{
    Strides original(Shape{2, 3, 4});
    Strides moved(std::move(original));

    EXPECT_TRUE(moved.rank() == 3);
    EXPECT_TRUE(moved(0) == 12);
    EXPECT_TRUE(moved(1) == 4);
    EXPECT_TRUE(moved(2) == 1);
}

TEST(CoreStrides, move_assignment)
{
    Strides original(Shape{2, 3, 4});
    Strides moved;

    moved = std::move(original);

    EXPECT_TRUE(moved.rank() == 3);
    EXPECT_TRUE(moved(0) == 12);
    EXPECT_TRUE(moved(1) == 4);
    EXPECT_TRUE(moved(2) == 1);

    Strides& same = moved;
    moved = std::move(same);
    EXPECT_TRUE(moved.rank() == 3);
    EXPECT_TRUE(moved(0) == 12);
    EXPECT_TRUE(moved(1) == 4);
    EXPECT_TRUE(moved(2) == 1);
}

TEST(CoreStrides, swap)
{
    Strides a(Shape{2, 3, 4});
    Strides b(Shape{5, 6});

    a.swap(b);

    EXPECT_TRUE(a.rank() == 2);
    EXPECT_TRUE(a(0) == 6);
    EXPECT_TRUE(a(1) == 1);

    EXPECT_TRUE(b.rank() == 3);
    EXPECT_TRUE(b(0) == 12);
    EXPECT_TRUE(b(1) == 4);
    EXPECT_TRUE(b(2) == 1);
}

// ============================================================================
// Const Method Coverage
// ============================================================================

TEST(CoreStrides, const_operator_parenthesis)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(strides(0) == 12);
    EXPECT_TRUE(strides(1) == 4);
    EXPECT_TRUE(strides(2) == 1);
}

TEST(CoreStrides, const_at)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(strides.at(0) == 12);
    EXPECT_TRUE(strides.at(1) == 4);
    EXPECT_TRUE(strides.at(2) == 1);

    bool threw = false;
    try {
        strides.at(3);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreStrides, const_front)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(strides.front() == 12);
}

TEST(CoreStrides, const_back)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(strides.back() == 1);
}

TEST(CoreStrides, const_data_pointer)
{
    const Strides strides(Shape{2, 3, 4});

    const std::size_t* ptr = strides.data();
    EXPECT_TRUE(ptr != nullptr);
    EXPECT_TRUE(*ptr == 12);
}

TEST(CoreStrides, const_data_pointer_empty)
{
    const Strides strides;

    EXPECT_TRUE(strides.data() == nullptr);
}

TEST(CoreStrides, const_rank)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(strides.rank() == 3);
}

TEST(CoreStrides, const_empty)
{
    const Strides empty;
    const Strides non_empty(Shape{1, 2, 3});

    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(!non_empty.empty());
}

TEST(CoreStrides, const_size)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(strides.size() == 3);
}

TEST(CoreStrides, const_equality)
{
    const Strides a(Shape{2, 3, 4});
    const Strides b(Shape{2, 3, 4});
    const Strides c(Shape{3, 4});

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a != c);
}

TEST(CoreStrides, const_begin_end)
{
    const Strides strides(Shape{2, 3, 4});

    const std::size_t* begin_ptr = strides.begin();
    const std::size_t* end_ptr = strides.end();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 12);
}

TEST(CoreStrides, const_cbegin_cend)
{
    const Strides strides(Shape{2, 3, 4});

    auto begin_ptr = strides.cbegin();
    auto end_ptr = strides.cend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
}

// ============================================================================
// Mutable Iterator Coverage
// ============================================================================

TEST(CoreStrides, mutable_begin_end)
{
    Strides strides(Shape{2, 3, 4});

    const std::size_t* begin_ptr = strides.begin();
    const std::size_t* end_ptr = strides.end();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 12);
}

TEST(CoreStrides, mutable_rbegin_rend)
{
    Strides strides(Shape{2, 3, 4});

    auto begin_ptr = strides.rbegin();
    auto end_ptr = strides.rend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 1);
}

TEST(CoreStrides, mutable_reverse_iteration)
{
    Strides strides(Shape{2, 3, 4});

    const std::size_t expected[] = {1, 4, 12};
    std::size_t i = 0;

    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
        EXPECT_TRUE(*it == expected[i]);
        ++i;
    }

    EXPECT_TRUE(i == 3);
}

TEST(CoreStrides, const_rbegin_rend)
{
    const Strides strides(Shape{2, 3, 4});

    auto begin_ptr = strides.rbegin();
    auto end_ptr = strides.rend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 1);
}

TEST(CoreStrides, const_crbegin_crend)
{
    const Strides strides(Shape{2, 3, 4});

    auto begin_ptr = strides.crbegin();
    auto end_ptr = strides.crend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
}

// ============================================================================
// Edge Cases & Error Conditions
// ============================================================================

TEST(CoreStrides, swap_with_empty)
{
    Strides filled(Shape{2, 3, 4});
    Strides empty;

    filled.swap(empty);

    EXPECT_TRUE(filled.empty());
    EXPECT_TRUE(filled.rank() == 0);
    EXPECT_TRUE(empty.rank() == 3);
    EXPECT_TRUE(empty(0) == 12);
}

TEST(CoreStrides, swap_empty_with_empty)
{
    Strides a;
    Strides b;

    a.swap(b);

    EXPECT_TRUE(a.empty());
    EXPECT_TRUE(b.empty());
}

TEST(CoreStrides, at_empty_throws)
{
    Strides strides;

    bool threw = false;
    try {
        strides.at(0);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreStrides, operator_parenthesis_out_of_bounds)
{
    Strides strides(Shape{2, 3, 4});

    // Note: operator() doesn't check bounds, so we can't test it throwing
    // This is by design (unchecked access)
}

TEST(CoreStrides, single_dimension)
{
    Strides strides(Shape{42});

    EXPECT_TRUE(strides.rank() == 1);
    EXPECT_TRUE(strides(0) == 1);
    EXPECT_TRUE(strides.front() == 1);
    EXPECT_TRUE(strides.back() == 1);
}

TEST(CoreStrides, many_dimensions)
{
    std::vector<std::size_t> dims(10, 2);
    Shape shape(dims);
    Strides strides(shape);

    EXPECT_TRUE(strides.rank() == 10);
    EXPECT_TRUE(strides(0) == 512);  // 2^9
    EXPECT_TRUE(strides(9) == 1);
}

TEST(CoreStrides, shape_with_all_ones)
{
    Strides strides(Shape{1, 1, 1, 1});

    EXPECT_TRUE(strides.rank() == 4);
    EXPECT_TRUE(strides(0) == 1);
    EXPECT_TRUE(strides(1) == 1);
    EXPECT_TRUE(strides(2) == 1);
    EXPECT_TRUE(strides(3) == 1);
}

TEST(CoreStrides, shape_with_leading_zeros)
{
    Strides strides(Shape{0, 0, 5});

    EXPECT_TRUE(strides.rank() == 3);
    EXPECT_TRUE(strides(0) == 0);
    EXPECT_TRUE(strides(1) == 5);
    EXPECT_TRUE(strides(2) == 1);
}

TEST(CoreStrides, inequality_operator)
{
    Strides a(Shape{2, 3, 4});
    Strides b(Shape{2, 3, 4});
    Strides c(Shape{3, 4});

    EXPECT_TRUE(!(a != b));
    EXPECT_TRUE(a != c);
}

// ============================================================================
// Iterator Consistency
// ============================================================================

TEST(CoreStrides, begin_end_distance)
{
    Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(std::distance(strides.begin(), strides.end()) == 3);
}

TEST(CoreStrides, cbegin_cend_distance)
{
    Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(std::distance(strides.cbegin(), strides.cend()) == 3);
}

TEST(CoreStrides, rbegin_rend_distance)
{
    Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(std::distance(strides.rbegin(), strides.rend()) == 3);
}

TEST(CoreStrides, crbegin_crend_distance)
{
    Strides strides(Shape{2, 3, 4});

    EXPECT_TRUE(std::distance(strides.crbegin(), strides.crend()) == 3);
}

TEST(CoreStrides, forward_and_backward_consistency)
{
    Strides strides(Shape{2, 3, 4});

    std::vector<std::size_t> forward_order;
    for (auto it = strides.begin(); it != strides.end(); ++it) {
        forward_order.push_back(*it);
    }

    std::vector<std::size_t> backward_order;
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
        backward_order.push_back(*it);
    }

    EXPECT_TRUE(forward_order.size() == backward_order.size());
    for (std::size_t i = 0; i < forward_order.size(); ++i) {
        EXPECT_TRUE(forward_order[i] == backward_order[backward_order.size() - 1 - i]);
    }
}

