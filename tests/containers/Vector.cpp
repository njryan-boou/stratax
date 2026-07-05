#include <cassert>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <stratax.hpp>

using namespace stratax::container;

void test_default_constructor()
{
    Vector<int> vector;

    assert(vector.empty());
    assert(vector.data() == nullptr);
    assert(vector.begin() == vector.end());
    assert(vector.shape().rank() == 0);
}

void test_size_constructor()
{
    Vector<int> vector(3);

    assert(vector.size() == 3);
    assert(!vector.empty());
    assert(vector.shape().rank() == 1);
    assert(vector.shape()(0) == 3);

    vector(0) = 10;
    vector(1) = 20;
    vector(2) = 30;

    assert(vector.front() == 10);
    assert(vector.back() == 30);
}

void test_fill_constructor()
{
    Vector<int> vector(4, 7);

    assert(vector.size() == 4);

    for (int value : vector) {
        assert(value == 7);
    }
}

void test_initializer_list_constructor()
{
    Vector<double> vector{1.5, 2.5, 3.5};

    assert(vector.size() == 3);
    assert(vector.shape().rank() == 1);
    assert(vector.shape()(0) == 3);
    assert(vector(0) == 1.5);
    assert(vector(1) == 2.5);
    assert(vector(2) == 3.5);
}

void test_at()
{
    Vector<int> vector{1, 2, 3};

    assert(vector.at(0) == 1);
    assert(vector.at(1) == 2);
    assert(vector.at(2) == 3);

    vector.at(1) = 20;
    assert(vector(1) == 20);

}

void test_const_access()
{
    const Vector<int> vector{4, 5, 6};

    assert(vector(0) == 4);
    assert(vector.at(1) == 5);
    assert(vector.front() == 4);
    assert(vector.back() == 6);
    assert(vector.data() != nullptr);
}

void test_iteration()
{
    Vector<int> vector{1, 2, 3, 4};

    const int sum = std::accumulate(vector.begin(), vector.end(), 0);

    assert(sum == 10);
}

void test_const_iteration()
{
    const Vector<int> vector{1, 2, 3};

    int expected = 1;

    for (auto it = vector.cbegin(); it != vector.cend(); ++it) {
        assert(*it == expected);
        ++expected;
    }

    assert(expected == 4);
}

void test_reverse_iteration()
{
    Vector<int> vector{1, 2, 3};

    int expected = 3;

    for (auto it = vector.rbegin(); it != vector.rend(); ++it) {
        assert(*it == expected);
        --expected;
    }

    assert(expected == 0);
}

void test_fill()
{
    Vector<int> vector{1, 2, 3};

    vector.fill(42);

    for (int value : vector) {
        assert(value == 42);
    }
}

void test_copy_constructor()
{
    Vector<int> original{1, 2, 3};
    Vector<int> copy(original);

    assert(copy.size() == original.size());
    assert(copy(0) == 1);
    assert(copy(1) == 2);
    assert(copy(2) == 3);

    copy(0) = 99;

    assert(original(0) == 1);
    assert(copy(0) == 99);
}

void test_copy_assignment()
{
    Vector<int> original{4, 5, 6};
    Vector<int> copy;

    copy = original;

    assert(copy.size() == 3);
    assert(copy(0) == 4);
    assert(copy(1) == 5);
    assert(copy(2) == 6);
}

void test_move_constructor()
{
    Vector<int> original{7, 8, 9};
    Vector<int> moved(std::move(original));

    assert(moved.size() == 3);
    assert(moved(0) == 7);
    assert(moved(1) == 8);
    assert(moved(2) == 9);
}

void test_move_assignment()
{
    Vector<int> original{10, 11};
    Vector<int> moved;

    moved = std::move(original);

    assert(moved.size() == 2);
    assert(moved(0) == 10);
    assert(moved(1) == 11);
}

void test_swap()
{
    Vector<int> a{1, 2};
    Vector<int> b{3, 4, 5};

    a.swap(b);

    assert(a.size() == 3);
    assert(a(0) == 3);
    assert(a(1) == 4);
    assert(a(2) == 5);

    assert(b.size() == 2);
    assert(b(0) == 1);
    assert(b(1) == 2);
}

int main()
{
    test_default_constructor();
    test_size_constructor();
    test_fill_constructor();
    test_initializer_list_constructor();
    test_at();
    test_const_access();
    test_iteration();
    test_const_iteration();
    test_reverse_iteration();
    test_fill();
    test_copy_constructor();
    test_copy_assignment();
    test_move_constructor();
    test_move_assignment();
    test_swap();

    return 0;
}
