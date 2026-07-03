#include <cassert>
#include <numeric>
#include <utility>

#include <stratax/containers/Tensor.hpp>

using namespace stratax::container;

void test_default_constructor()
{
    Tensor<int> tensor;

    assert(tensor.size() == 0);
    assert(tensor.empty());
    assert(tensor.rank() == 0);
    assert(tensor.data() == nullptr);
    assert(tensor.begin() == tensor.end());
}

void test_shape_constructor()
{
    Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    assert(tensor.size() == 24);
    assert(!tensor.empty());
    assert(tensor.rank() == 3);

    assert(tensor.shape()[0] == 2);
    assert(tensor.shape()[1] == 3);
    assert(tensor.shape()[2] == 4);

    assert(tensor.strides()[0] == 12);
    assert(tensor.strides()[1] == 4);
    assert(tensor.strides()[2] == 1);
}

void test_fill_constructor()
{
    Tensor<int> tensor(stratax::core::Shape{2, 3}, 7);

    assert(tensor.size() == 6);
    assert(tensor.rank() == 2);

    for (int value : tensor) {
        assert(value == 7);
    }
}

void test_flat_element_access()
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    assert(tensor[0] == 1);
    assert(tensor[1] == 2);
    assert(tensor[2] == 3);
    assert(tensor[3] == 4);
    assert(tensor[4] == 5);
    assert(tensor[5] == 6);

    assert(tensor.front() == 1);
    assert(tensor.back() == 6);
}

void test_at()
{
    Tensor<int> tensor(stratax::core::Shape{2, 2});

    tensor.at(0) = 10;
    tensor.at(1) = 20;
    tensor.at(2) = 30;
    tensor.at(3) = 40;

    assert(tensor.at(0) == 10);
    assert(tensor.at(1) == 20);
    assert(tensor.at(2) == 30);
    assert(tensor.at(3) == 40);

    bool threw = false;

    try {
        tensor.at(4);
    }
    catch (const stratax::core::IndexError&) {
        threw = true;
    }

    assert(threw);
}

void test_const_access()
{
    const Tensor<int> tensor(stratax::core::Shape{2, 2}, 5);

    assert(tensor[0] == 5);
    assert(tensor.at(1) == 5);
    assert(tensor.front() == 5);
    assert(tensor.back() == 5);
    assert(tensor.data() != nullptr);
}

void test_iteration()
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    const int sum = std::accumulate(tensor.begin(), tensor.end(), 0);

    assert(sum == 21);
}

void test_const_iteration()
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3}, 4);

    std::size_t count = 0;

    for (auto it = tensor.cbegin(); it != tensor.cend(); ++it) {
        assert(*it == 4);
        ++count;
    }

    assert(count == 6);
}

void test_reverse_iteration()
{
    Tensor<int> tensor(stratax::core::Shape{3});

    tensor[0] = 1;
    tensor[1] = 2;
    tensor[2] = 3;

    int expected = 3;

    for (auto it = tensor.rbegin(); it != tensor.rend(); ++it) {
        assert(*it == expected);
        --expected;
    }

    assert(expected == 0);
}

void test_fill()
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    tensor.fill(42);

    for (int value : tensor) {
        assert(value == 42);
    }
}

void test_copy_constructor()
{
    Tensor<int> original(stratax::core::Shape{2, 2});

    original[0] = 1;
    original[1] = 2;
    original[2] = 3;
    original[3] = 4;

    Tensor<int> copy(original);

    assert(copy.size() == original.size());
    assert(copy.rank() == original.rank());
    assert(copy[0] == 1);
    assert(copy[1] == 2);
    assert(copy[2] == 3);
    assert(copy[3] == 4);

    copy[0] = 99;

    assert(original[0] == 1);
    assert(copy[0] == 99);
}

void test_copy_assignment()
{
    Tensor<int> original(stratax::core::Shape{2, 2}, 8);
    Tensor<int> copy;

    copy = original;

    assert(copy.size() == 4);
    assert(copy.rank() == 2);

    for (int value : copy) {
        assert(value == 8);
    }
}

void test_move_constructor()
{
    Tensor<int> original(stratax::core::Shape{2, 2}, 9);
    Tensor<int> moved(std::move(original));

    assert(moved.size() == 4);
    assert(moved.rank() == 2);

    for (int value : moved) {
        assert(value == 9);
    }
}

void test_move_assignment()
{
    Tensor<int> original(stratax::core::Shape{2, 3}, 10);
    Tensor<int> moved;

    moved = std::move(original);

    assert(moved.size() == 6);
    assert(moved.rank() == 2);

    for (int value : moved) {
        assert(value == 10);
    }
}

void test_swap()
{
    Tensor<int> a(stratax::core::Shape{2}, 1);
    Tensor<int> b(stratax::core::Shape{2, 2}, 2);

    a.swap(b);

    assert(a.size() == 4);
    assert(a.rank() == 2);
    assert(a.shape()[0] == 2);
    assert(a.shape()[1] == 2);

    for (int value : a) {
        assert(value == 2);
    }

    assert(b.size() == 2);
    assert(b.rank() == 1);
    assert(b.shape()[0] == 2);

    for (int value : b) {
        assert(value == 1);
    }
}

int main()
{
    test_default_constructor();
    test_shape_constructor();
    test_fill_constructor();
    test_flat_element_access();
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
