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

