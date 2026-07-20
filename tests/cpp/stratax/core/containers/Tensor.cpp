#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <stratax.h>

using namespace stratax::container;

TEST(ContainersTensor, default_constructor)
{
    Tensor<int> tensor;

    EXPECT_TRUE(tensor.size() == 0);
    EXPECT_TRUE(tensor.empty());
    EXPECT_TRUE(tensor.rank() == 0);
    EXPECT_TRUE(tensor.data() == nullptr);
    EXPECT_TRUE(tensor.begin() == tensor.end());

    bool front_threw = false;
    try {
        tensor.front();
    }
    catch (const Exceptions::IndexError&) {
        front_threw = true;
    }

    EXPECT_TRUE(front_threw);

    bool back_threw = false;
    try {
        tensor.back();
    }
    catch (const Exceptions::IndexError&) {
        back_threw = true;
    }

    EXPECT_TRUE(back_threw);
}

TEST(ContainersTensor, shape_constructor)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    EXPECT_TRUE(tensor.size() == 24);
    EXPECT_TRUE(!tensor.empty());
    EXPECT_TRUE(tensor.rank() == 3);

    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.shape()(2) == 4);

    EXPECT_TRUE(tensor.strides()(0) == 12);
    EXPECT_TRUE(tensor.strides()(1) == 4);
    EXPECT_TRUE(tensor.strides()(2) == 1);
}

TEST(ContainersTensor, empty_shape_constructor)
{
    Tensor<int> tensor(stratax::core::Shape{});

    EXPECT_TRUE(tensor.size() == 0);
    EXPECT_TRUE(tensor.empty());
    EXPECT_TRUE(tensor.rank() == 0);
    EXPECT_TRUE(tensor.shape().empty());
    EXPECT_TRUE(tensor.strides().empty());
}

TEST(ContainersTensor, zero_dimension_shape_constructor)
{
    Tensor<int> tensor(stratax::core::Shape{2, 0, 4});

    EXPECT_TRUE(tensor.size() == 0);
    EXPECT_TRUE(tensor.empty());
    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 0);
    EXPECT_TRUE(tensor.shape()(2) == 4);
    EXPECT_TRUE(tensor.strides()(0) == 0);
    EXPECT_TRUE(tensor.strides()(1) == 4);
    EXPECT_TRUE(tensor.strides()(2) == 1);
}

TEST(ContainersTensor, shape_overflow_throws)
{
    EXPECT_THROW(
        Tensor<int> tensor(stratax::core::Shape{2, std::numeric_limits<long long>::max(), 2}),
        Exceptions::StrataxError);
}

TEST(ContainersTensor, fill_constructor)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3}, 7);

    EXPECT_TRUE(tensor.size() == 6);
    EXPECT_TRUE(tensor.rank() == 2);

    for (int value : tensor) {
        EXPECT_TRUE(value == 7);
    }
}

TEST(ContainersTensor, flat_element_access)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i + 1);
    }

    EXPECT_TRUE(tensor(0) == 1);
    EXPECT_TRUE(tensor(1) == 2);
    EXPECT_TRUE(tensor(2) == 3);
    EXPECT_TRUE(tensor(3) == 4);
    EXPECT_TRUE(tensor(4) == 5);
    EXPECT_TRUE(tensor(5) == 6);

    EXPECT_TRUE(tensor.front() == 1);
    EXPECT_TRUE(tensor.back() == 6);
}

TEST(ContainersTensor, linear_index_operator)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    EXPECT_TRUE(tensor[0] == 1);
    EXPECT_TRUE(tensor[1] == 2);
    EXPECT_TRUE(tensor[2] == 3);
    EXPECT_TRUE(tensor[3] == 4);
    EXPECT_TRUE(tensor[4] == 5);
    EXPECT_TRUE(tensor[5] == 6);

    tensor[4] = 50;
    EXPECT_TRUE(tensor(1, 1) == 50);

    const Tensor<int>& view = tensor;
    EXPECT_TRUE(view[4] == 50);
}

TEST(ContainersTensor, multi_index_access)
{
    Tensor<int> matrix_like(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < matrix_like.size(); ++i) {
        matrix_like(i) = static_cast<int>(i + 1);
    }

    EXPECT_TRUE(matrix_like(0, 0) == 1);
    EXPECT_TRUE(matrix_like(0, 2) == 3);
    EXPECT_TRUE(matrix_like(1, 0) == 4);
    EXPECT_TRUE(matrix_like(1, 2) == 6);

    matrix_like(1, 1) = 40;
    EXPECT_TRUE(matrix_like(4) == 40);

    Tensor<int> cube(stratax::core::Shape{2, 3, 4});

    for (std::size_t i = 0; i < cube.size(); ++i) {
        cube(i) = static_cast<int>(i);
    }

    EXPECT_TRUE(cube(0, 0, 0) == 0);
    EXPECT_TRUE(cube(0, 1, 2) == 6);
    EXPECT_TRUE(cube(1, 0, 2) == 14);
    EXPECT_TRUE(cube(1, 2, 3) == 23);
}

TEST(ContainersTensor, const_multi_index_access)
{
    Tensor<int> tensor(stratax::core::Shape{2, 2});

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;
    tensor(3) = 4;

    const Tensor<int>& view = tensor;

    EXPECT_TRUE(view(0, 0) == 1);
    EXPECT_TRUE(view(0, 1) == 2);
    EXPECT_TRUE(view(1, 0) == 3);
    EXPECT_TRUE(view(1, 1) == 4);
}

TEST(ContainersTensor, at)
{
    Tensor<int> tensor(stratax::core::Shape{2, 2});

    tensor.at(0) = 10;
    tensor.at(1) = 20;
    tensor.at(2) = 30;
    tensor.at(3) = 40;

    EXPECT_TRUE(tensor.at(0) == 10);
    EXPECT_TRUE(tensor.at(1) == 20);
    EXPECT_TRUE(tensor.at(2) == 30);
    EXPECT_TRUE(tensor.at(3) == 40);

    tensor.at(1, 1) = 50;
    EXPECT_TRUE(tensor.at(1, 1) == 50);
    EXPECT_TRUE(tensor.at(3) == 50);

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

    EXPECT_TRUE(flat_threw);
    EXPECT_TRUE(row_threw);
    EXPECT_TRUE(col_threw);
    EXPECT_TRUE(rank_threw);
    EXPECT_TRUE(one_dim_rank_threw);
}

TEST(ContainersTensor, at_supports_negative_index)
{
    Tensor<int> tensor(stratax::core::Shape{2, 2});

    tensor.at(-1) = 40;
    EXPECT_TRUE(tensor.at(-1) == 40);

    tensor.at(0, 0) = 10;
    tensor.at(-1, -2) = 30;

    EXPECT_TRUE(tensor.at(-2, -2) == 10);
    EXPECT_TRUE(tensor.at(-1, -2) == 30);

    bool flat_threw = false;
    bool multi_threw = false;

    try {
        [[maybe_unused]] auto value = tensor.at(-5);
    }
    catch (const Exceptions::IndexError&) {
        flat_threw = true;
    }

    try {
        [[maybe_unused]] auto value = tensor.at(-3, 0);
    }
    catch (const Exceptions::IndexError&) {
        multi_threw = true;
    }

    EXPECT_TRUE(flat_threw);
    EXPECT_TRUE(multi_threw);
}

TEST(ContainersTensor, const_at_rejects_out_of_bounds)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 2}, 5);

    EXPECT_TRUE(tensor.at(0) == 5);
    EXPECT_TRUE(tensor.at(1, 1) == 5);

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

    EXPECT_TRUE(flat_threw);
    EXPECT_TRUE(multi_threw);
}

TEST(ContainersTensor, const_access)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 2}, 5);

    EXPECT_TRUE(tensor(0) == 5);
    EXPECT_TRUE(tensor.at(1) == 5);
    EXPECT_TRUE(tensor.front() == 5);
    EXPECT_TRUE(tensor.back() == 5);
    EXPECT_TRUE(tensor.data() != nullptr);
}

TEST(ContainersTensor, iteration)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i + 1);
    }

    const int sum = std::accumulate(tensor.begin(), tensor.end(), 0);

    EXPECT_TRUE(sum == 21);
}

TEST(ContainersTensor, const_iteration)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3}, 4);

    std::size_t count = 0;

    for (auto it = tensor.cbegin(); it != tensor.cend(); ++it) {
        EXPECT_TRUE(*it == 4);
        ++count;
    }

    EXPECT_TRUE(count == 6);
}

TEST(ContainersTensor, reverse_iteration)
{
    Tensor<int> tensor(stratax::core::Shape{3});

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;

    int expected = 3;

    for (auto it = tensor.rbegin(); it != tensor.rend(); ++it) {
        EXPECT_TRUE(*it == expected);
        --expected;
    }

    EXPECT_TRUE(expected == 0);
}

TEST(ContainersTensor, fill)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    tensor.fill(42);

    for (int value : tensor) {
        EXPECT_TRUE(value == 42);
    }

    Tensor<int> empty(stratax::core::Shape{0});
    empty.fill(7);
    EXPECT_TRUE(empty.empty());
}

TEST(ContainersTensor, copy_constructor)
{
    Tensor<int> original(stratax::core::Shape{2, 2});

    original(0) = 1;
    original(1) = 2;
    original(2) = 3;
    original(3) = 4;

    Tensor<int> copy(original);

    EXPECT_TRUE(copy.size() == original.size());
    EXPECT_TRUE(copy.rank() == original.rank());
    EXPECT_TRUE(copy(0) == 1);
    EXPECT_TRUE(copy(1) == 2);
    EXPECT_TRUE(copy(2) == 3);
    EXPECT_TRUE(copy(3) == 4);

    copy(0) = 99;

    EXPECT_TRUE(original(0) == 1);
    EXPECT_TRUE(copy(0) == 99);
}

TEST(ContainersTensor, copy_assignment)
{
    Tensor<int> original(stratax::core::Shape{2, 2}, 8);
    Tensor<int> copy;

    copy = original;

    EXPECT_TRUE(copy.size() == 4);
    EXPECT_TRUE(copy.rank() == 2);

    for (int value : copy) {
        EXPECT_TRUE(value == 8);
    }

    copy = copy;
    EXPECT_TRUE(copy.size() == 4);
    EXPECT_TRUE(copy.rank() == 2);

    for (int value : copy) {
        EXPECT_TRUE(value == 8);
    }
}

TEST(ContainersTensor, move_constructor)
{
    Tensor<int> original(stratax::core::Shape{2, 2}, 9);
    Tensor<int> moved(std::move(original));

    EXPECT_TRUE(moved.size() == 4);
    EXPECT_TRUE(moved.rank() == 2);

    for (int value : moved) {
        EXPECT_TRUE(value == 9);
    }
}

TEST(ContainersTensor, move_assignment)
{
    Tensor<int> original(stratax::core::Shape{2, 3}, 10);
    Tensor<int> moved;

    moved = std::move(original);

    EXPECT_TRUE(moved.size() == 6);
    EXPECT_TRUE(moved.rank() == 2);

    for (int value : moved) {
        EXPECT_TRUE(value == 10);
    }

    Tensor<int>& same = moved;
    moved = std::move(same);
    EXPECT_TRUE(moved.size() == 6);
    EXPECT_TRUE(moved.rank() == 2);

    for (int value : moved) {
        EXPECT_TRUE(value == 10);
    }
}

TEST(ContainersTensor, swap)
{
    Tensor<int> a(stratax::core::Shape{2}, 1);
    Tensor<int> b(stratax::core::Shape{2, 2}, 2);

    a.swap(b);

    EXPECT_TRUE(a.size() == 4);
    EXPECT_TRUE(a.rank() == 2);
    EXPECT_TRUE(a.shape()(0) == 2);
    EXPECT_TRUE(a.shape()(1) == 2);

    for (int value : a) {
        EXPECT_TRUE(value == 2);
    }

    EXPECT_TRUE(b.size() == 2);
    EXPECT_TRUE(b.rank() == 1);
    EXPECT_TRUE(b.shape()(0) == 2);

    for (int value : b) {
        EXPECT_TRUE(value == 1);
    }
}

TEST(ContainersTensor, swap_with_empty)
{
    Tensor<int> populated(stratax::core::Shape{2, 2}, 5);
    Tensor<int> empty;

    populated.swap(empty);

    EXPECT_TRUE(populated.empty());
    EXPECT_TRUE(populated.rank() == 0);
    EXPECT_TRUE(empty.size() == 4);
    EXPECT_TRUE(empty.rank() == 2);
    EXPECT_TRUE(empty.shape()(0) == 2);
    EXPECT_TRUE(empty.shape()(1) == 2);
    EXPECT_TRUE(empty.strides()(0) == 2);
    EXPECT_TRUE(empty.strides()(1) == 1);

    for (int value : empty) {
        EXPECT_TRUE(value == 5);
    }
}

// ============================================================================
// Const Method Coverage
// ============================================================================

TEST(ContainersTensor, const_operator_parenthesis_flat)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3});

    EXPECT_TRUE(tensor(0) == 0);
    EXPECT_TRUE(tensor(5) == 0);
}

TEST(ContainersTensor, const_operator_bracket)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3});

    EXPECT_TRUE(tensor[0] == 0);
    EXPECT_TRUE(tensor[5] == 0);
}

TEST(ContainersTensor, const_operator_parenthesis_multidim)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i);
    }

    const Tensor<int>& view = tensor;

    EXPECT_TRUE(view(0, 0, 0) == 0);
    EXPECT_TRUE(view(0, 1, 2) == 6);
    EXPECT_TRUE(view(1, 0, 2) == 14);
    EXPECT_TRUE(view(1, 2, 3) == 23);
}

TEST(ContainersTensor, const_front)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3}, 42);

    EXPECT_TRUE(tensor.front() == 42);
}

TEST(ContainersTensor, const_back)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3}, 99);

    EXPECT_TRUE(tensor.back() == 99);
}

TEST(ContainersTensor, const_data)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3});

    const int* ptr = tensor.data();
    EXPECT_TRUE(ptr != nullptr);
    EXPECT_TRUE(*ptr == 0);
}

TEST(ContainersTensor, const_data_empty)
{
    const Tensor<int> tensor;

    EXPECT_TRUE(tensor.data() == nullptr);
}

TEST(ContainersTensor, const_size)
{
    const Tensor<int> tensor(stratax::core::Shape{3, 4, 5});

    EXPECT_TRUE(tensor.size() == 60);
}

TEST(ContainersTensor, const_rank)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    EXPECT_TRUE(tensor.rank() == 3);
}

TEST(ContainersTensor, const_empty)
{
    const Tensor<int> empty;
    const Tensor<int> non_empty(stratax::core::Shape{2, 2});

    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(!non_empty.empty());
}

TEST(ContainersTensor, const_shape)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    const auto& shape = tensor.shape();
    EXPECT_TRUE(shape.rank() == 3);
    EXPECT_TRUE(shape(0) == 2);
    EXPECT_TRUE(shape(1) == 3);
    EXPECT_TRUE(shape(2) == 4);
}

TEST(ContainersTensor, const_strides)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    const auto& strides = tensor.strides();
    EXPECT_TRUE(strides.rank() == 3);
    EXPECT_TRUE(strides(0) == 12);
    EXPECT_TRUE(strides(1) == 4);
    EXPECT_TRUE(strides(2) == 1);
}

// ============================================================================
// Mutable Iterator Coverage
// ============================================================================

TEST(ContainersTensor, mutable_begin_end)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    auto begin_ptr = tensor.begin();
    auto end_ptr = tensor.end();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 6);
}

TEST(ContainersTensor, mutable_rbegin_rend)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    auto begin_ptr = tensor.rbegin();
    auto end_ptr = tensor.rend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 6);
}

TEST(ContainersTensor, mutable_iteration_modification)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (auto it = tensor.begin(); it != tensor.end(); ++it) {
        *it = 42;
    }

    for (int value : tensor) {
        EXPECT_TRUE(value == 42);
    }
}

TEST(ContainersTensor, mutable_reverse_iteration_modification)
{
    Tensor<int> tensor(stratax::core::Shape{3});

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;

    for (auto it = tensor.rbegin(); it != tensor.rend(); ++it) {
        *it *= 10;
    }

    EXPECT_TRUE(tensor(0) == 10);
    EXPECT_TRUE(tensor(1) == 20);
    EXPECT_TRUE(tensor(2) == 30);
}

TEST(ContainersTensor, const_begin_end)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3});

    auto begin_ptr = tensor.begin();
    auto end_ptr = tensor.end();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 6);
}

TEST(ContainersTensor, const_cbegin_cend)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3});

    auto begin_ptr = tensor.cbegin();
    auto end_ptr = tensor.cend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 6);
}

TEST(ContainersTensor, const_rbegin_rend)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3});

    auto begin_ptr = tensor.rbegin();
    auto end_ptr = tensor.rend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 6);
}

TEST(ContainersTensor, const_crbegin_crend)
{
    const Tensor<int> tensor(stratax::core::Shape{2, 3});

    auto begin_ptr = tensor.crbegin();
    auto end_ptr = tensor.crend();

    EXPECT_TRUE(std::distance(begin_ptr, end_ptr) == 6);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ContainersTensor, single_element_tensor)
{
    Tensor<int> tensor(stratax::core::Shape{1}, 42);

    EXPECT_TRUE(tensor.size() == 1);
    EXPECT_TRUE(tensor.rank() == 1);
    EXPECT_TRUE(tensor(0) == 42);
    EXPECT_TRUE(tensor.front() == 42);
    EXPECT_TRUE(tensor.back() == 42);
}

TEST(ContainersTensor, single_element_multidim_tensor)
{
    Tensor<int> tensor(stratax::core::Shape{1, 1, 1}, 99);

    EXPECT_TRUE(tensor.size() == 1);
    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor(0, 0, 0) == 99);
}

TEST(ContainersTensor, large_tensor)
{
    Tensor<int> tensor(stratax::core::Shape{10, 10, 10});

    EXPECT_TRUE(tensor.size() == 1000);
    EXPECT_TRUE(tensor.rank() == 3);

    tensor(500) = 999;
    EXPECT_TRUE(tensor(500) == 999);
}

TEST(ContainersTensor, high_rank_tensor)
{
    Tensor<int> tensor(stratax::core::Shape{2, 2, 2, 2, 2, 2});

    EXPECT_TRUE(tensor.size() == 64);
    EXPECT_TRUE(tensor.rank() == 6);

    // Initialize with sequential values
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i);
    }

    EXPECT_TRUE(tensor(0, 0, 0, 0, 0, 0) == 0);
    EXPECT_TRUE(tensor(1, 1, 1, 1, 1, 1) == 63);
}

TEST(ContainersTensor, zero_dimension_tensor)
{
    Tensor<int> tensor(stratax::core::Shape{2, 0, 3});

    EXPECT_TRUE(tensor.empty());
    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.shape()(1) == 0);
}

TEST(ContainersTensor, rank_one_tensor)
{
    Tensor<int> tensor(stratax::core::Shape{10}, 5);

    EXPECT_TRUE(tensor.size() == 10);
    EXPECT_TRUE(tensor.rank() == 1);

    for (int value : tensor) {
        EXPECT_TRUE(value == 5);
    }
}

TEST(ContainersTensor, high_rank_strides)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3, 4, 5});

    EXPECT_TRUE(tensor.strides()(0) == 60);
    EXPECT_TRUE(tensor.strides()(1) == 20);
    EXPECT_TRUE(tensor.strides()(2) == 5);
    EXPECT_TRUE(tensor.strides()(3) == 1);
}

// ============================================================================
// Error Conditions
// ============================================================================

TEST(ContainersTensor, front_throws_on_empty)
{
    Tensor<int> tensor;

    bool threw = false;
    try {
        tensor.front();
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersTensor, back_throws_on_empty)
{
    Tensor<int> tensor;

    bool threw = false;
    try {
        tensor.back();
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersTensor, flat_at_out_of_bounds)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    bool threw = false;
    try {
        tensor.at(6);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(ContainersTensor, multiindex_at_out_of_bounds)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    bool row_threw = false;
    bool col_threw = false;
    bool depth_threw = false;

    try {
        tensor.at(3, 0, 0);
    }
    catch (const Exceptions::IndexError&) {
        row_threw = true;
    }

    try {
        tensor.at(0, 3, 0);
    }
    catch (const Exceptions::IndexError&) {
        col_threw = true;
    }

    try {
        tensor.at(0, 0, 4);
    }
    catch (const Exceptions::IndexError&) {
        depth_threw = true;
    }

    EXPECT_TRUE(row_threw);
    EXPECT_TRUE(col_threw);
    EXPECT_TRUE(depth_threw);
}

TEST(ContainersTensor, at_negative_out_of_bounds)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    bool flat_threw = false;
    bool multi_threw = false;

    try {
        tensor.at(-10);
    }
    catch (const Exceptions::IndexError&) {
        flat_threw = true;
    }

    try {
        tensor.at(-3, 0);
    }
    catch (const Exceptions::IndexError&) {
        multi_threw = true;
    }

    EXPECT_TRUE(flat_threw);
    EXPECT_TRUE(multi_threw);
}

// ============================================================================
// Resource Management
// ============================================================================

TEST(ContainersTensor, copy_assignment_overwrites_old_data)
{
    Tensor<int> source(stratax::core::Shape{2, 2}, 5);
    Tensor<int> target(stratax::core::Shape{3, 3, 3}, 99);

    target = source;

    EXPECT_TRUE(target.size() == 4);
    EXPECT_TRUE(target.rank() == 2);
    EXPECT_TRUE(target.shape()(0) == 2);
    EXPECT_TRUE(target.shape()(1) == 2);

    for (int value : target) {
        EXPECT_TRUE(value == 5);
    }
}

TEST(ContainersTensor, move_assignment_on_nonempty)
{
    Tensor<int> source(stratax::core::Shape{2, 3}, 10);
    Tensor<int> target(stratax::core::Shape{1, 1}, 99);

    target = std::move(source);

    EXPECT_TRUE(target.size() == 6);
    EXPECT_TRUE(target.rank() == 2);
    EXPECT_TRUE(target.shape()(0) == 2);
    EXPECT_TRUE(target.shape()(1) == 3);

    for (int value : target) {
        EXPECT_TRUE(value == 10);
    }
}

// ============================================================================
// Iterator Consistency
// ============================================================================

TEST(ContainersTensor, begin_end_distance_consistency)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    EXPECT_TRUE(std::distance(tensor.begin(), tensor.end()) == 24);
    EXPECT_TRUE(std::distance(tensor.cbegin(), tensor.cend()) == 24);
}

TEST(ContainersTensor, rbegin_rend_distance_consistency)
{
    Tensor<int> tensor(stratax::core::Shape{3, 4, 5});

    EXPECT_TRUE(std::distance(tensor.rbegin(), tensor.rend()) == 60);
    EXPECT_TRUE(std::distance(tensor.crbegin(), tensor.crend()) == 60);
}

TEST(ContainersTensor, forward_and_backward_consistency)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i + 1);
    }

    std::vector<int> forward_order;
    for (auto it = tensor.begin(); it != tensor.end(); ++it) {
        forward_order.push_back(*it);
    }

    std::vector<int> backward_order;
    for (auto it = tensor.rbegin(); it != tensor.rend(); ++it) {
        backward_order.push_back(*it);
    }

    EXPECT_TRUE(forward_order.size() == backward_order.size());
    for (std::size_t i = 0; i < forward_order.size(); ++i) {
        EXPECT_TRUE(forward_order[i] == backward_order[backward_order.size() - 1 - i]);
    }
}

TEST(ContainersTensor, swap_empty_with_empty)
{
    Tensor<int> a;
    Tensor<int> b;

    a.swap(b);

    EXPECT_TRUE(a.empty());
    EXPECT_TRUE(b.empty());
}

TEST(ContainersTensor, data_pointer_consistency)
{
    Tensor<int> tensor(stratax::core::Shape{2, 3, 4});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i);
    }

    int* ptr = tensor.data();
    EXPECT_TRUE(ptr == tensor.begin());
    EXPECT_TRUE(ptr[0] == tensor(0));
    EXPECT_TRUE(ptr[23] == tensor(23));
}

// ============================================================================
// Complex Type Testing
// ============================================================================

TEST(ContainersTensor, tensor_of_doubles)
{
    Tensor<double> tensor(stratax::core::Shape{2, 3}, 3.14);

    EXPECT_TRUE(tensor.size() == 6);
    EXPECT_TRUE(tensor.front() == 3.14);
    EXPECT_TRUE(tensor.back() == 3.14);
}

TEST(ContainersTensor, flat_multi_index_equivalence)
{
    Tensor<int> tensor(stratax::core::Shape{3, 4, 5});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i);
    }

    // Access (1, 2, 3) should be at flat index 1*20 + 2*5 + 3 = 33
    EXPECT_TRUE(tensor(1, 2, 3) == tensor(33));
    EXPECT_TRUE(tensor(1, 2, 3) == 33);
}

