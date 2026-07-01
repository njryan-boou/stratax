#include <cassert>
#include <numeric>
#include <utility>

#include <stratax/containers/Matrix.hpp>

using namespace stratax::container;

void test_default_constructor()
{
    Matrix<int> matrix;

    assert(matrix.empty());
    assert(matrix.data() == nullptr);
    assert(matrix.begin() == matrix.end());
    assert(matrix.shape().rank() == 0);
}

void test_shape_constructor()
{
    Matrix<int> matrix(2, 3);

    assert(matrix.size() == 6);
    assert(!matrix.empty());
    assert(matrix.rows() == 2);
    assert(matrix.cols() == 3);
    assert(matrix.shape().rank() == 2);
    assert(matrix.shape()[0] == 2);
    assert(matrix.shape()[1] == 3);

    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(0, 2) = 3;
    matrix(1, 0) = 4;
    matrix(1, 1) = 5;
    matrix(1, 2) = 6;

    assert(matrix.front() == 1);
    assert(matrix.back() == 6);
}

void test_fill_constructor()
{
    Matrix<int> matrix(2, 2, 7);

    assert(matrix.size() == 4);
    assert(matrix.rows() == 2);
    assert(matrix.cols() == 2);

    for (int value : matrix) {
        assert(value == 7);
    }
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

    assert(matrix(0, 0) == 1);
    assert(matrix(0, 1) == 2);
    assert(matrix(0, 2) == 3);
    assert(matrix(1, 0) == 4);
    assert(matrix(1, 1) == 5);
    assert(matrix(1, 2) == 6);
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
    catch (const stratax::core::StrataxError&) {
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
    catch (const stratax::core::IndexError&) {
        row_threw = true;
    }

    try {
        matrix.at(0, 2);
    }
    catch (const stratax::core::IndexError&) {
        col_threw = true;
    }

    assert(row_threw);
    assert(col_threw);
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
    Matrix<int> copy;

    copy = original;

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
    Matrix<int> moved;

    moved = std::move(original);

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

int main()
{
    test_default_constructor();
    test_shape_constructor();
    test_fill_constructor();
    test_initializer_list_constructor();
    test_initializer_list_rejects_ragged_rows();
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