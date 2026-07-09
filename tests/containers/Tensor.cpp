#include <cassert>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <stratax.hpp>

using namespace stratax::container;

void test_default_constructor()
{
    Tensor<int> tensor;

    assert(tensor.size() == 0);
    assert(tensor.empty());
    assert(tensor.rank() == 0);
    assert(tensor.data() == nullptr);
    assert(tensor.begin() == tensor.end());

    bool front_threw = false;
    try {
        tensor.front();
    }
    catch (const Exceptions::IndexError&) {
        front_threw = true;
    }

    assert(front_threw);

    bool back_threw = false;
    try {
        tensor.back();
    }
    catch (const Exceptions::IndexError&) {
        back_threw = true;
    }

    assert(back_threw);
}

void test_shape_constructor()
{
    Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    assert(tensor.size() == 24);
    assert(!tensor.empty());
    assert(tensor.rank() == 3);

    assert(tensor.shape()(0) == 2);
    assert(tensor.shape()(1) == 3);
    assert(tensor.shape()(2) == 4);

    assert(tensor.strides()(0) == 12);
    assert(tensor.strides()(1) == 4);
    assert(tensor.strides()(2) == 1);
}

void test_empty_shape_constructor()
{
    Tensor<int> tensor(stratax::core::Shape{});

    assert(tensor.size() == 0);
    assert(tensor.empty());
    assert(tensor.rank() == 0);
    assert(tensor.shape().empty());
    assert(tensor.strides().empty());
}

void test_zero_dimension_shape_constructor()
{
    Tensor<int> tensor(stratax::core::Shape{2, 0, 4});

    assert(tensor.size() == 0);
    assert(tensor.empty());
    assert(tensor.rank() == 3);
    assert(tensor.shape()(0) == 2);
    assert(tensor.shape()(1) == 0);
    assert(tensor.shape()(2) == 4);
    assert(tensor.strides()(0) == 0);
    assert(tensor.strides()(1) == 4);
    assert(tensor.strides()(2) == 1);
}

void test_shape_overflow_throws()
{
    bool threw = false;

    try {
        Tensor<int> tensor(stratax::core::Shape{2, std::numeric_limits<std::size_t>::max(), 2});
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    assert(threw);
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
        tensor(i) = static_cast<int>(i + 1);
    }

    assert(tensor(0) == 1);
    assert(tensor(1) == 2);
    assert(tensor(2) == 3);
    assert(tensor(3) == 4);
    assert(tensor(4) == 5);
    assert(tensor(5) == 6);

    assert(tensor.front() == 1);
    assert(tensor.back() == 6);
}

void test_linear_index_operator()
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

    tensor[4] = 50;
    assert(tensor(1, 1) == 50);

    const Tensor<int>& view = tensor;
    assert(view[4] == 50);
}

void test_multi_index_access()
{
    Tensor<int> matrix_like(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < matrix_like.size(); ++i) {
        matrix_like(i) = static_cast<int>(i + 1);
    }

    assert(matrix_like(0, 0) == 1);
    assert(matrix_like(0, 2) == 3);
    assert(matrix_like(1, 0) == 4);
    assert(matrix_like(1, 2) == 6);

    matrix_like(1, 1) = 40;
    assert(matrix_like(4) == 40);

    Tensor<int> cube(stratax::core::Shape{2, 3, 4});

    for (std::size_t i = 0; i < cube.size(); ++i) {
        cube(i) = static_cast<int>(i);
    }

    assert(cube(0, 0, 0) == 0);
    assert(cube(0, 1, 2) == 6);
    assert(cube(1, 0, 2) == 14);
    assert(cube(1, 2, 3) == 23);
}

void test_const_multi_index_access()
{
    Tensor<int> tensor(stratax::core::Shape{2, 2});

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;
    tensor(3) = 4;

    const Tensor<int>& view = tensor;

    assert(view(0, 0) == 1);
    assert(view(0, 1) == 2);
    assert(view(1, 0) == 3);
    assert(view(1, 1) == 4);
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

    tensor.at(1, 1) = 50;
    assert(tensor.at(1, 1) == 50);
    assert(tensor.at(3) == 50);

    bool flat_threw = false;
    bool row_threw = false;
    bool col_threw = false;
    bool rank_threw = false;
    bool one_dim_rank_threw = false;

    try {
        tensor.at(4);
    }
    catch (const Exceptions::IndexError&) {
        flat_threw = true;
    }

    try {
        tensor.at(2, 0);
    }
    catch (const Exceptions::IndexError&) {
        row_threw = true;
    }

    try {
        tensor.at(0, 2);
    }
    catch (const Exceptions::IndexError&) {
        col_threw = true;
    }

    try {
        tensor.at(0, 0, 0);
    }
    catch (const Exceptions::IndexError&) {
        rank_threw = true;
    }

    Tensor<int> vector_like(stratax::core::Shape{3}, 1);

    try {
        vector_like.at(0, 0);
    }
    catch (const Exceptions::IndexError&) {
        one_dim_rank_threw = true;
    }

    assert(flat_threw);
    assert(row_threw);
    assert(col_threw);
    assert(rank_threw);
    assert(one_dim_rank_threw);
}

void test_const_at_rejects_out_of_bounds()
{
    const Tensor<int> tensor(stratax::core::Shape{2, 2}, 5);

    assert(tensor.at(0) == 5);
    assert(tensor.at(1, 1) == 5);

    bool flat_threw = false;
    bool multi_threw = false;

    try {
        tensor.at(4);
    }
    catch (const Exceptions::IndexError&) {
        flat_threw = true;
    }

    try {
        tensor.at(1, 2);
    }
    catch (const Exceptions::IndexError&) {
        multi_threw = true;
    }

    assert(flat_threw);
    assert(multi_threw);
}

void test_const_access()
{
    const Tensor<int> tensor(stratax::core::Shape{2, 2}, 5);

    assert(tensor(0) == 5);
    assert(tensor.at(1) == 5);
    assert(tensor.front() == 5);
    assert(tensor.back() == 5);
    assert(tensor.data() != nullptr);
}

void test_iteration()
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i + 1);
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

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;

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

    Tensor<int> empty(stratax::core::Shape{0});
    empty.fill(7);
    assert(empty.empty());
}

void test_copy_constructor()
{
    Tensor<int> original(stratax::core::Shape{2, 2});

    original(0) = 1;
    original(1) = 2;
    original(2) = 3;
    original(3) = 4;

    Tensor<int> copy(original);

    assert(copy.size() == original.size());
    assert(copy.rank() == original.rank());
    assert(copy(0) == 1);
    assert(copy(1) == 2);
    assert(copy(2) == 3);
    assert(copy(3) == 4);

    copy(0) = 99;

    assert(original(0) == 1);
    assert(copy(0) == 99);
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

    copy = copy;
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

    Tensor<int>& same = moved;
    moved = std::move(same);
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
    assert(a.shape()(0) == 2);
    assert(a.shape()(1) == 2);

    for (int value : a) {
        assert(value == 2);
    }

    assert(b.size() == 2);
    assert(b.rank() == 1);
    assert(b.shape()(0) == 2);

    for (int value : b) {
        assert(value == 1);
    }
}

void test_swap_with_empty()
{
    Tensor<int> populated(stratax::core::Shape{2, 2}, 5);
    Tensor<int> empty;

    populated.swap(empty);

    assert(populated.empty());
    assert(populated.rank() == 0);
    assert(empty.size() == 4);
    assert(empty.rank() == 2);
    assert(empty.shape()(0) == 2);
    assert(empty.shape()(1) == 2);
    assert(empty.strides()(0) == 2);
    assert(empty.strides()(1) == 1);

    for (int value : empty) {
        assert(value == 5);
    }
}

int main()
{
    test_default_constructor();
    test_shape_constructor();
    test_empty_shape_constructor();
    test_zero_dimension_shape_constructor();
    test_shape_overflow_throws();
    test_fill_constructor();
    test_flat_element_access();
    test_linear_index_operator();
    test_multi_index_access();
    test_const_multi_index_access();
    test_at();
    test_const_at_rejects_out_of_bounds();
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
    test_swap_with_empty();

    return 0;
}
