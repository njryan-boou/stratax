#include <cassert>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <stratax.hpp>

using namespace stratax::container;

void test_default_constructor()
{
    Matrix<int> matrix(0, 0);

    assert(matrix.empty());
    assert(matrix.data() == nullptr);
    assert(matrix.begin() == matrix.end());
    assert(matrix.shape().rank() == 2);
    assert(matrix.rows() == 0);
    assert(matrix.cols() == 0);
    assert(matrix.strides().rank() == 2);
    assert(matrix.strides()(0) == 0);
    assert(matrix.strides()(1) == 1);

    bool front_threw = false;
    try {
        matrix.front();
    }
    catch (const Exceptions::IndexError&) {
        front_threw = true;
    }

    assert(front_threw);

    bool back_threw = false;
    try {
        matrix.back();
    }
    catch (const Exceptions::IndexError&) {
        back_threw = true;
    }

    assert(back_threw);
}

void test_fill_constructor()
{
    Matrix<int> matrix(2, 2, 7);

    assert(matrix.size() == 4);
    assert(matrix.rows() == 2);
    assert(matrix.cols() == 2);
    assert(matrix.strides().rank() == 2);
    assert(matrix.strides()(0) == 2);
    assert(matrix.strides()(1) == 1);

    for (int value : matrix) {
        assert(value == 7);
    }
}

void test_shape_constructor()
{
    Matrix<int> matrix(stratax::core::Shape{2, 3});

    assert(matrix.size() == 6);
    assert(matrix.rows() == 2);
    assert(matrix.cols() == 3);
    assert(matrix.shape().rank() == 2);
    assert(matrix.strides().rank() == 2);
    assert(matrix.strides()(0) == 3);
    assert(matrix.strides()(1) == 1);
}

void test_shape_constructor_rejects_wrong_rank()
{
    bool rank_one_threw = false;
    try {
        Matrix<int> matrix(stratax::core::Shape{3});
    }
    catch (const Exceptions::DimensionError&) {
        rank_one_threw = true;
    }

    assert(rank_one_threw);

    bool rank_three_threw = false;
    try {
        Matrix<int> matrix(stratax::core::Shape{2, 3, 4});
    }
    catch (const Exceptions::DimensionError&) {
        rank_three_threw = true;
    }

    assert(rank_three_threw);
}

void test_size_overflow_throws()
{
    bool direct_threw = false;
    try {
        Matrix<int> matrix(std::numeric_limits<std::size_t>::max(), 2);
    }
    catch (const Exceptions::DimensionError&) {
        direct_threw = true;
    }

    assert(direct_threw);

    bool shape_threw = false;
    try {
        Matrix<int> matrix(stratax::core::Shape{std::numeric_limits<std::size_t>::max(), 2});
    }
    catch (const Exceptions::DimensionError&) {
        shape_threw = true;
    }

    assert(shape_threw);
}

void test_initializer_list_constructor()
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    assert(matrix.size() == 6);
    assert(matrix.rows() == 2);
    assert(matrix.cols() == 3);
    assert(matrix.strides().rank() == 2);
    assert(matrix.strides()(0) == 3);
    assert(matrix.strides()(1) == 1);

    assert(matrix(0, 0) == 1);
    assert(matrix(0, 1) == 2);
    assert(matrix(0, 2) == 3);
    assert(matrix(1, 0) == 4);
    assert(matrix(1, 1) == 5);
    assert(matrix(1, 2) == 6);
}

void test_empty_initializer_list_constructor()
{
    Matrix<int> matrix(std::initializer_list<std::initializer_list<int>>{});

    assert(matrix.empty());
    assert(matrix.size() == 0);
    assert(matrix.rows() == 0);
    assert(matrix.cols() == 0);
    assert(matrix.shape().rank() == 2);
    assert(matrix.strides().rank() == 2);
    assert(matrix.strides()(0) == 0);
    assert(matrix.strides()(1) == 1);
}

void test_initializer_list_rejects_ragged_rows()
{
    bool threw = false;

    try {
        Matrix<int> matrix{
            {1, 2},
            {3}
        };
    }
    catch (const Exceptions::StrataxError&) {
        threw = true;
    }

    assert(threw);
}

void test_at()
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    assert(matrix.at(0, 0) == 1);
    assert(matrix.at(0, 1) == 2);
    assert(matrix.at(1, 0) == 3);
    assert(matrix.at(1, 1) == 4);

    matrix.at(1, 1) = 40;
    assert(matrix(1, 1) == 40);

    bool row_threw = false;
    bool col_threw = false;

    try {
        matrix.at(2, 0);
    }
    catch (const Exceptions::IndexError&) {
        row_threw = true;
    }

    try {
        matrix.at(0, 2);
    }
    catch (const Exceptions::IndexError&) {
        col_threw = true;
    }

    assert(row_threw);
    assert(col_threw);
}

void test_linear_index_operator()
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    assert(matrix[0] == 1);
    assert(matrix[1] == 2);
    assert(matrix[2] == 3);
    assert(matrix[3] == 4);
    assert(matrix[4] == 5);
    assert(matrix[5] == 6);

    matrix[4] = 50;
    assert(matrix(1, 1) == 50);

    const Matrix<int>& view = matrix;
    assert(view[4] == 50);
}

void test_const_access()
{
    const Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    assert(matrix(0, 0) == 1);
    assert(matrix.at(1, 1) == 4);
    assert(matrix.front() == 1);
    assert(matrix.back() == 4);
    assert(matrix.data() != nullptr);
}

void test_iteration()
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    const int sum = std::accumulate(matrix.begin(), matrix.end(), 0);

    assert(sum == 10);
}

void test_const_iteration()
{
    const Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    int expected = 1;

    for (auto it = matrix.cbegin(); it != matrix.cend(); ++it) {
        assert(*it == expected);
        ++expected;
    }

    assert(expected == 5);
}

void test_reverse_iteration()
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    int expected = 4;

    for (auto it = matrix.rbegin(); it != matrix.rend(); ++it) {
        assert(*it == expected);
        --expected;
    }

    assert(expected == 0);
}

void test_fill()
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    matrix.fill(42);

    for (int value : matrix) {
        assert(value == 42);
    }

    Matrix<int> empty(0, 0);
    empty.fill(7);
    assert(empty.empty());
}

void test_copy_constructor()
{
    Matrix<int> original{
        {1, 2},
        {3, 4}
    };
    Matrix<int> copy(original);

    assert(copy.rows() == 2);
    assert(copy.cols() == 2);
    assert(copy(0, 0) == 1);
    assert(copy(1, 1) == 4);

    copy(0, 0) = 99;

    assert(original(0, 0) == 1);
    assert(copy(0, 0) == 99);
}

void test_copy_assignment()
{
    Matrix<int> original{
        {5, 6},
        {7, 8}
    };
    Matrix<int> copy(0, 0);

    copy = original;

    assert(copy.rows() == 2);
    assert(copy.cols() == 2);
    assert(copy(0, 0) == 5);
    assert(copy(1, 1) == 8);

    copy = copy;
    assert(copy.rows() == 2);
    assert(copy.cols() == 2);
    assert(copy(0, 0) == 5);
    assert(copy(1, 1) == 8);
}

void test_move_constructor()
{
    Matrix<int> original{
        {1, 2, 3},
        {4, 5, 6}
    };
    Matrix<int> moved(std::move(original));

    assert(moved.rows() == 2);
    assert(moved.cols() == 3);
    assert(moved(0, 0) == 1);
    assert(moved(1, 2) == 6);
}

void test_move_assignment()
{
    Matrix<int> original{
        {9, 10},
        {11, 12}
    };
    Matrix<int> moved(0, 0);

    moved = std::move(original);

    assert(moved.rows() == 2);
    assert(moved.cols() == 2);
    assert(moved(0, 0) == 9);
    assert(moved(1, 1) == 12);

    Matrix<int>& same = moved;
    moved = std::move(same);
    assert(moved.rows() == 2);
    assert(moved.cols() == 2);
    assert(moved(0, 0) == 9);
    assert(moved(1, 1) == 12);
}

void test_swap()
{
    Matrix<int> a{
        {1, 2}
    };
    Matrix<int> b{
        {3, 4},
        {5, 6}
    };

    a.swap(b);

    assert(a.rows() == 2);
    assert(a.cols() == 2);
    assert(a(0, 0) == 3);
    assert(a(1, 1) == 6);

    assert(b.rows() == 1);
    assert(b.cols() == 2);
    assert(b(0, 0) == 1);
    assert(b(0, 1) == 2);
}

void test_swap_with_empty()
{
    Matrix<int> populated{
        {1, 2},
        {3, 4}
    };
    Matrix<int> empty(0, 0);

    populated.swap(empty);

    assert(populated.empty());
    assert(empty.rows() == 2);
    assert(empty.cols() == 2);
    assert(empty(0, 0) == 1);
    assert(empty(1, 1) == 4);
    assert(empty.strides().rank() == 2);
    assert(empty.strides()(0) == 2);
    assert(empty.strides()(1) == 1);
}

int main()
{
    test_default_constructor();
    test_fill_constructor();
    test_shape_constructor();
    test_shape_constructor_rejects_wrong_rank();
    test_size_overflow_throws();
    test_initializer_list_constructor();
    test_empty_initializer_list_constructor();
    test_initializer_list_rejects_ragged_rows();
    test_at();
    test_linear_index_operator();
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
