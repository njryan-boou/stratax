#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <stratax.h>

using namespace stratax::container;

TEST(ContainersVector, default_constructor)
{
    Vector<int> vector;

    EXPECT_TRUE(vector.empty());
    EXPECT_TRUE(vector.data() == nullptr);
    EXPECT_TRUE(vector.begin() == vector.end());
    EXPECT_TRUE(vector.shape().rank() == 0);

    bool front_threw = false;
    try {
        vector.front();
    }
    catch (const Exceptions::IndexError&) {
        front_threw = true;
    }

    EXPECT_TRUE(front_threw);

    bool back_threw = false;
    try {
        vector.back();
    }
    catch (const Exceptions::IndexError&) {
        back_threw = true;
    }

    EXPECT_TRUE(back_threw);
}

TEST(ContainersVector, size_constructor)
{
    Vector<int> vector(3);

    EXPECT_TRUE(vector.size() == 3);
    EXPECT_TRUE(!vector.empty());
    EXPECT_TRUE(vector.shape().rank() == 1);
    EXPECT_TRUE(vector.shape()(0) == 3);

    vector(0) = 10;
    vector(1) = 20;
    vector(2) = 30;

    EXPECT_TRUE(vector.front() == 10);
    EXPECT_TRUE(vector.back() == 30);
}

TEST(ContainersVector, fill_constructor)
{
    Vector<int> vector(4, 7);

    EXPECT_TRUE(vector.size() == 4);

    for (int value : vector) {
        EXPECT_TRUE(value == 7);
    }
}

TEST(ContainersVector, initializer_list_constructor)
{
    Vector<double> vector{1.5, 2.5, 3.5};

    EXPECT_TRUE(vector.size() == 3);
    EXPECT_TRUE(vector.shape().rank() == 1);
    EXPECT_TRUE(vector.shape()(0) == 3);
    EXPECT_TRUE(vector(0) == 1.5);
    EXPECT_TRUE(vector(1) == 2.5);
    EXPECT_TRUE(vector(2) == 3.5);
}

TEST(ContainersVector, empty_initializer_list_constructor)
{
    Vector<int> vector(std::initializer_list<int>{});

    EXPECT_TRUE(vector.empty());
    EXPECT_TRUE(vector.size() == 0);
    EXPECT_TRUE(vector.shape().rank() == 1);
    EXPECT_TRUE(vector.shape()(0) == 0);
    EXPECT_TRUE(vector.strides().rank() == 1);
    EXPECT_TRUE(vector.strides()(0) == 1);
}

TEST(ContainersVector, shape_constructor)
{
    Vector<int> vector(stratax::core::Shape{3});

    EXPECT_TRUE(vector.size() == 3);
    EXPECT_TRUE(vector.shape().rank() == 1);
    EXPECT_TRUE(vector.shape()(0) == 3);
    EXPECT_TRUE(vector.strides().rank() == 1);
    EXPECT_TRUE(vector.strides()(0) == 1);
}

TEST(ContainersVector, shape_constructor_rejects_wrong_rank)
{
    bool rank_zero_threw = false;
    try {
        Vector<int> vector(stratax::core::Shape{});
    }
    catch (const Exceptions::DimensionError&) {
        rank_zero_threw = true;
    }

    EXPECT_TRUE(rank_zero_threw);

    bool rank_two_threw = false;
    try {
        Vector<int> vector(stratax::core::Shape{2, 3});
    }
    catch (const Exceptions::DimensionError&) {
        rank_two_threw = true;
    }

    EXPECT_TRUE(rank_two_threw);
}

TEST(ContainersVector, at)
{
    Vector<int> vector{1, 2, 3};

    EXPECT_TRUE(vector.at(0) == 1);
    EXPECT_TRUE(vector.at(1) == 2);
    EXPECT_TRUE(vector.at(2) == 3);

    vector.at(1) = 20;
    EXPECT_TRUE(vector(1) == 20);

    bool threw = false;
    try {
        vector.at(3);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);

}

TEST(ContainersVector, at_supports_negative_index)
{
    Vector<int> vector{1, 2, 3};

    EXPECT_TRUE(vector.at(-1) == 3);
    EXPECT_TRUE(vector.at(-3) == 1);

    vector.at(-2) = 20;
    EXPECT_TRUE(vector(1) == 20);

    bool threw = false;
    try {
        [[maybe_unused]] auto value = vector.at(-4);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersVector, linear_index_operator)
{
    Vector<int> vector{1, 2, 3};

    EXPECT_TRUE(vector[0] == 1);
    EXPECT_TRUE(vector[1] == 2);
    EXPECT_TRUE(vector[2] == 3);

    vector[1] = 20;
    EXPECT_TRUE(vector(1) == 20);

    const Vector<int>& view = vector;
    EXPECT_TRUE(view[1] == 20);
}

TEST(ContainersVector, const_access)
{
    const Vector<int> vector{4, 5, 6};

    EXPECT_TRUE(vector(0) == 4);
    EXPECT_TRUE(vector.at(1) == 5);
    EXPECT_TRUE(vector.front() == 4);
    EXPECT_TRUE(vector.back() == 6);
    EXPECT_TRUE(vector.data() != nullptr);
}

TEST(ContainersVector, iteration)
{
    Vector<int> vector{1, 2, 3, 4};

    const int sum = std::accumulate(vector.begin(), vector.end(), 0);

    EXPECT_TRUE(sum == 10);
}

TEST(ContainersVector, const_iteration)
{
    const Vector<int> vector{1, 2, 3};

    int expected = 1;

    for (auto it = vector.cbegin(); it != vector.cend(); ++it) {
        EXPECT_TRUE(*it == expected);
        ++expected;
    }

    EXPECT_TRUE(expected == 4);
}

TEST(ContainersVector, reverse_iteration)
{
    Vector<int> vector{1, 2, 3};

    int expected = 3;

    for (auto it = vector.rbegin(); it != vector.rend(); ++it) {
        EXPECT_TRUE(*it == expected);
        --expected;
    }

    EXPECT_TRUE(expected == 0);
}

TEST(ContainersVector, fill)
{
    Vector<int> vector{1, 2, 3};

    vector.fill(42);

    for (int value : vector) {
        EXPECT_TRUE(value == 42);
    }

    Vector<int> empty;
    empty.fill(7);
    EXPECT_TRUE(empty.empty());
}

TEST(ContainersVector, copy_constructor)
{
    Vector<int> original{1, 2, 3};
    Vector<int> copy(original);

    EXPECT_TRUE(copy.size() == original.size());
    EXPECT_TRUE(copy(0) == 1);
    EXPECT_TRUE(copy(1) == 2);
    EXPECT_TRUE(copy(2) == 3);

    copy(0) = 99;

    EXPECT_TRUE(original(0) == 1);
    EXPECT_TRUE(copy(0) == 99);
}

TEST(ContainersVector, copy_assignment)
{
    Vector<int> original{4, 5, 6};
    Vector<int> copy;

    copy = original;

    EXPECT_TRUE(copy.size() == 3);
    EXPECT_TRUE(copy(0) == 4);
    EXPECT_TRUE(copy(1) == 5);
    EXPECT_TRUE(copy(2) == 6);

    copy = copy;
    EXPECT_TRUE(copy.size() == 3);
    EXPECT_TRUE(copy(0) == 4);
    EXPECT_TRUE(copy(1) == 5);
    EXPECT_TRUE(copy(2) == 6);
}

TEST(ContainersVector, move_constructor)
{
    Vector<int> original{7, 8, 9};
    Vector<int> moved(std::move(original));

    EXPECT_TRUE(moved.size() == 3);
    EXPECT_TRUE(moved(0) == 7);
    EXPECT_TRUE(moved(1) == 8);
    EXPECT_TRUE(moved(2) == 9);
}

TEST(ContainersVector, move_assignment)
{
    Vector<int> original{10, 11};
    Vector<int> moved;

    moved = std::move(original);

    EXPECT_TRUE(moved.size() == 2);
    EXPECT_TRUE(moved(0) == 10);
    EXPECT_TRUE(moved(1) == 11);

    Vector<int>& same = moved;
    moved = std::move(same);
    EXPECT_TRUE(moved.size() == 2);
    EXPECT_TRUE(moved(0) == 10);
    EXPECT_TRUE(moved(1) == 11);
}

TEST(ContainersVector, swap)
{
    Vector<int> a{1, 2};
    Vector<int> b{3, 4, 5};

    a.swap(b);

    EXPECT_TRUE(a.size() == 3);
    EXPECT_TRUE(a(0) == 3);
    EXPECT_TRUE(a(1) == 4);
    EXPECT_TRUE(a(2) == 5);

    EXPECT_TRUE(b.size() == 2);
    EXPECT_TRUE(b(0) == 1);
    EXPECT_TRUE(b(1) == 2);
}

TEST(ContainersVector, swap_with_empty)
{
    Vector<int> populated{1, 2, 3};
    Vector<int> empty;

    populated.swap(empty);

    EXPECT_TRUE(populated.empty());
    EXPECT_TRUE(empty.size() == 3);
    EXPECT_TRUE(empty(0) == 1);
    EXPECT_TRUE(empty(1) == 2);
    EXPECT_TRUE(empty(2) == 3);
    EXPECT_TRUE(empty.shape().rank() == 1);
    EXPECT_TRUE(empty.strides().rank() == 1);
}

// ============================================================================
// Const Method Coverage
// ============================================================================

TEST(ContainersVector, const_operator_parenthesis)
{
    const Vector<int> vector{10, 20, 30};

    EXPECT_TRUE(vector(0) == 10);
    EXPECT_TRUE(vector(1) == 20);
    EXPECT_TRUE(vector(2) == 30);
}

TEST(ContainersVector, const_operator_bracket)
{
    const Vector<int> vector{5, 6, 7};

    EXPECT_TRUE(vector[0] == 5);
    EXPECT_TRUE(vector[1] == 6);
    EXPECT_TRUE(vector[2] == 7);
}

TEST(ContainersVector, const_at)
{
    const Vector<int> vector{1, 2, 3};

    EXPECT_TRUE(vector.at(0) == 1);
    EXPECT_TRUE(vector.at(-1) == 3);
    EXPECT_TRUE(vector.at(-2) == 2);
}

TEST(ContainersVector, const_at_throws_out_of_bounds)
{
    const Vector<int> vector{1, 2, 3};

    bool threw = false;
    try {
        vector.at(10);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersVector, const_front)
{
    const Vector<int> vector{42, 43, 44};

    EXPECT_TRUE(vector.front() == 42);
}

TEST(ContainersVector, const_back)
{
    const Vector<int> vector{42, 43, 44};

    EXPECT_TRUE(vector.back() == 44);
}

TEST(ContainersVector, const_data)
{
    const Vector<int> vector{1, 2, 3};

    const int* ptr = vector.data();
    EXPECT_TRUE(ptr != nullptr);
    EXPECT_TRUE(*ptr == 1);
}

TEST(ContainersVector, const_data_empty)
{
    const Vector<int> vector;

    EXPECT_TRUE(vector.data() == nullptr);
}

TEST(ContainersVector, const_size)
{
    const Vector<int> vector{1, 2, 3, 4, 5};

    EXPECT_TRUE(vector.size() == 5);
}

TEST(ContainersVector, const_rank)
{
    const Vector<int> vector{1, 2, 3};

    EXPECT_TRUE(vector.rank() == 1);
}

TEST(ContainersVector, const_rank_default_vector)
{
    const Vector<int> vector;

    EXPECT_TRUE(vector.rank() == 0);
}

TEST(ContainersVector, const_empty)
{
    const Vector<int> empty;
    const Vector<int> non_empty{1, 2, 3};

    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(!non_empty.empty());
}

TEST(ContainersVector, const_shape)
{
    const Vector<int> vector{1, 2, 3};

    const auto& shape = vector.shape();
    EXPECT_TRUE(shape.rank() == 1);
    EXPECT_TRUE(shape(0) == 3);
}

TEST(ContainersVector, const_strides)
{
    const Vector<int> vector{1, 2, 3};

    const auto& strides = vector.strides();
    EXPECT_TRUE(strides.rank() == 1);
    EXPECT_TRUE(strides(0) == 1);
}

// ============================================================================
// Mutable Iterator Coverage
// ============================================================================

TEST(ContainersVector, mutable_begin_end)
{
    Vector<int> vector{1, 2, 3};

    int* begin_ptr = vector.begin();
    int* end_ptr = vector.end();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 1);
}

TEST(ContainersVector, mutable_rbegin_rend)
{
    Vector<int> vector{1, 2, 3};

    auto begin_ptr = vector.rbegin();
    auto end_ptr = vector.rend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 3);
}

TEST(ContainersVector, mutable_iteration_modification)
{
    Vector<int> vector{1, 2, 3};

    for (auto it = vector.begin(); it != vector.end(); ++it) {
        *it *= 10;
    }

    EXPECT_TRUE(vector(0) == 10);
    EXPECT_TRUE(vector(1) == 20);
    EXPECT_TRUE(vector(2) == 30);
}

TEST(ContainersVector, mutable_reverse_iteration_modification)
{
    Vector<int> vector{1, 2, 3};

    for (auto it = vector.rbegin(); it != vector.rend(); ++it) {
        *it *= 10;
    }

    EXPECT_TRUE(vector(0) == 10);
    EXPECT_TRUE(vector(1) == 20);
    EXPECT_TRUE(vector(2) == 30);
}

TEST(ContainersVector, const_begin_end)
{
    const Vector<int> vector{1, 2, 3};

    const int* begin_ptr = vector.begin();
    const int* end_ptr = vector.end();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 1);
}

TEST(ContainersVector, const_cbegin_cend)
{
    const Vector<int> vector{1, 2, 3};

    auto begin_ptr = vector.cbegin();
    auto end_ptr = vector.cend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
}

TEST(ContainersVector, const_rbegin_rend)
{
    const Vector<int> vector{1, 2, 3};

    auto begin_ptr = vector.rbegin();
    auto end_ptr = vector.rend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
    EXPECT_TRUE(*begin_ptr == 3);
}

TEST(ContainersVector, const_crbegin_crend)
{
    const Vector<int> vector{1, 2, 3};

    auto begin_ptr = vector.crbegin();
    auto end_ptr = vector.crend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 3);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ContainersVector, single_element_vector)
{
    Vector<int> vector(1);

    EXPECT_TRUE(vector.size() == 1);
    EXPECT_TRUE(!vector.empty());
    EXPECT_TRUE(vector.front() == vector.back());
    EXPECT_TRUE(vector.shape()(0) == 1);
}

TEST(ContainersVector, large_vector)
{
    Vector<int> vector(1000);

    EXPECT_TRUE(vector.size() == 1000);
    EXPECT_TRUE(vector.shape()(0) == 1000);

    vector(500) = 999;
    EXPECT_TRUE(vector(500) == 999);
}

TEST(ContainersVector, zero_size_vector)
{
    Vector<int> vector(0);

    EXPECT_TRUE(vector.empty());
    EXPECT_TRUE(vector.size() == 0);
    EXPECT_TRUE(vector.shape().rank() == 1);
    EXPECT_TRUE(vector.shape()(0) == 0);
}

// ============================================================================
// Error Conditions
// ============================================================================

TEST(ContainersVector, front_throws_on_empty)
{
    Vector<int> vector;

    bool threw = false;
    try {
        vector.front();
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersVector, back_throws_on_empty)
{
    Vector<int> vector;

    bool threw = false;
    try {
        vector.back();
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersVector, at_negative_out_of_bounds)
{
    Vector<int> vector{1, 2, 3};

    bool threw = false;
    try {
        vector.at(-10);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersVector, fill_on_empty)
{
    Vector<int> vector(0);

    vector.fill(42);

    EXPECT_TRUE(vector.empty());
}

// ============================================================================
// Resource Management
// ============================================================================

TEST(ContainersVector, copy_assignment_overwrites_old_data)
{
    Vector<int> source{1, 2, 3};
    Vector<int> target{99, 99, 99, 99, 99};

    target = source;

    EXPECT_TRUE(target.size() == 3);
    EXPECT_TRUE(target(0) == 1);
    EXPECT_TRUE(target(1) == 2);
    EXPECT_TRUE(target(2) == 3);
}

TEST(ContainersVector, move_assignment_on_nonempty)
{
    Vector<int> source{10, 20, 30};
    Vector<int> target{99, 99};

    target = std::move(source);

    EXPECT_TRUE(target.size() == 3);
    EXPECT_TRUE(target(0) == 10);
    EXPECT_TRUE(target(1) == 20);
    EXPECT_TRUE(target(2) == 30);
}

// ============================================================================
// Iterator Consistency
// ============================================================================

TEST(ContainersVector, begin_end_distance)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    EXPECT_TRUE(std::distance(vector.begin(), vector.end()) == 5);
}

TEST(ContainersVector, cbegin_cend_distance)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    EXPECT_TRUE(std::distance(vector.cbegin(), vector.cend()) == 5);
}

TEST(ContainersVector, rbegin_rend_distance)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    EXPECT_TRUE(std::distance(vector.rbegin(), vector.rend()) == 5);
}

TEST(ContainersVector, crbegin_crend_distance)
{
    Vector<int> vector{1, 2, 3, 4, 5};

    EXPECT_TRUE(std::distance(vector.crbegin(), vector.crend()) == 5);
}

TEST(ContainersVector, forward_and_backward_consistency)
{
    Vector<int> vector{10, 20, 30, 40};

    std::vector<int> forward_order;
    for (auto it = vector.begin(); it != vector.end(); ++it) {
        forward_order.push_back(*it);
    }

    std::vector<int> backward_order;
    for (auto it = vector.rbegin(); it != vector.rend(); ++it) {
        backward_order.push_back(*it);
    }

    EXPECT_TRUE(forward_order.size() == backward_order.size());
    for (std::size_t i = 0; i < forward_order.size(); ++i) {
        EXPECT_TRUE(forward_order[i] == backward_order[backward_order.size() - 1 - i]);
    }
}

TEST(ContainersVector, swap_empty_with_empty)
{
    Vector<int> a;
    Vector<int> b;

    a.swap(b);

    EXPECT_TRUE(a.empty());
    EXPECT_TRUE(b.empty());
}

TEST(ContainersVector, data_pointer_consistency)
{
    Vector<int> vector{5, 6, 7};

    int* ptr = vector.data();
    EXPECT_TRUE(ptr == vector.begin());
    EXPECT_TRUE(ptr[0] == vector(0));
    EXPECT_TRUE(ptr[2] == vector(2));
}

