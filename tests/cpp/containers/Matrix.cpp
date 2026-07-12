#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <stratax.h>

using namespace stratax::container;

TEST(ContainersMatrix, default_constructor)
{
    Matrix<int> matrix(0, 0);

    EXPECT_TRUE(matrix.empty());
    EXPECT_TRUE(matrix.data() == nullptr);
    EXPECT_TRUE(matrix.begin() == matrix.end());
    EXPECT_TRUE(matrix.shape().rank() == 2);
    EXPECT_TRUE(matrix.rows() == 0);
    EXPECT_TRUE(matrix.cols() == 0);
    EXPECT_TRUE(matrix.strides().rank() == 2);
    EXPECT_TRUE(matrix.strides()(0) == 0);
    EXPECT_TRUE(matrix.strides()(1) == 1);

    bool front_threw = false;
    try {
        matrix.front();
    }
    catch (const Exceptions::IndexError&) {
        front_threw = true;
    }

    EXPECT_TRUE(front_threw);

    bool back_threw = false;
    try {
        matrix.back();
    }
    catch (const Exceptions::IndexError&) {
        back_threw = true;
    }

    EXPECT_TRUE(back_threw);
}

TEST(ContainersMatrix, fill_constructor)
{
    Matrix<int> matrix(2, 2, 7);

    EXPECT_TRUE(matrix.size() == 4);
    EXPECT_TRUE(matrix.rows() == 2);
    EXPECT_TRUE(matrix.cols() == 2);
    EXPECT_TRUE(matrix.strides().rank() == 2);
    EXPECT_TRUE(matrix.strides()(0) == 2);
    EXPECT_TRUE(matrix.strides()(1) == 1);

    for (int value : matrix) {
        EXPECT_TRUE(value == 7);
    }
}

TEST(ContainersMatrix, shape_constructor)
{
    Matrix<int> matrix(stratax::core::Shape{2, 3});

    EXPECT_TRUE(matrix.size() == 6);
    EXPECT_TRUE(matrix.rows() == 2);
    EXPECT_TRUE(matrix.cols() == 3);
    EXPECT_TRUE(matrix.shape().rank() == 2);
    EXPECT_TRUE(matrix.strides().rank() == 2);
    EXPECT_TRUE(matrix.strides()(0) == 3);
    EXPECT_TRUE(matrix.strides()(1) == 1);
}

TEST(ContainersMatrix, shape_constructor_rejects_wrong_rank)
{
    bool rank_one_threw = false;
    try {
        Matrix<int> matrix(stratax::core::Shape{3});
    }
    catch (const Exceptions::DimensionError&) {
        rank_one_threw = true;
    }

    EXPECT_TRUE(rank_one_threw);

    bool rank_three_threw = false;
    try {
        Matrix<int> matrix(stratax::core::Shape{2, 3, 4});
    }
    catch (const Exceptions::DimensionError&) {
        rank_three_threw = true;
    }

    EXPECT_TRUE(rank_three_threw);
}

TEST(ContainersMatrix, size_overflow_throws)
{
    EXPECT_THROW(
        Matrix<int> matrix(std::numeric_limits<std::size_t>::max(), 2),
        Exceptions::StrataxError);

    EXPECT_THROW(
        Matrix<int> matrix(stratax::core::Shape{std::numeric_limits<long long>::max(), 2}),
        std::exception);
}

TEST(ContainersMatrix, initializer_list_constructor)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    EXPECT_TRUE(matrix.size() == 6);
    EXPECT_TRUE(matrix.rows() == 2);
    EXPECT_TRUE(matrix.cols() == 3);
    EXPECT_TRUE(matrix.strides().rank() == 2);
    EXPECT_TRUE(matrix.strides()(0) == 3);
    EXPECT_TRUE(matrix.strides()(1) == 1);

    EXPECT_TRUE(matrix(0, 0) == 1);
    EXPECT_TRUE(matrix(0, 1) == 2);
    EXPECT_TRUE(matrix(0, 2) == 3);
    EXPECT_TRUE(matrix(1, 0) == 4);
    EXPECT_TRUE(matrix(1, 1) == 5);
    EXPECT_TRUE(matrix(1, 2) == 6);
}

TEST(ContainersMatrix, empty_initializer_list_constructor)
{
    Matrix<int> matrix(std::initializer_list<std::initializer_list<int>>{});

    EXPECT_TRUE(matrix.empty());
    EXPECT_TRUE(matrix.size() == 0);
    EXPECT_TRUE(matrix.rows() == 0);
    EXPECT_TRUE(matrix.cols() == 0);
    EXPECT_TRUE(matrix.shape().rank() == 2);
    EXPECT_TRUE(matrix.strides().rank() == 2);
    EXPECT_TRUE(matrix.strides()(0) == 0);
    EXPECT_TRUE(matrix.strides()(1) == 1);
}

TEST(ContainersMatrix, initializer_list_rejects_ragged_rows)
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

    EXPECT_TRUE(threw);
}

TEST(ContainersMatrix, at)
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    EXPECT_TRUE(matrix.at(0, 0) == 1);
    EXPECT_TRUE(matrix.at(0, 1) == 2);
    EXPECT_TRUE(matrix.at(1, 0) == 3);
    EXPECT_TRUE(matrix.at(1, 1) == 4);

    matrix.at(1, 1) = 40;
    EXPECT_TRUE(matrix(1, 1) == 40);

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

    EXPECT_TRUE(row_threw);
    EXPECT_TRUE(col_threw);
}

TEST(ContainersMatrix, linear_index_operator)
{
    Matrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    EXPECT_TRUE(matrix[0] == 1);
    EXPECT_TRUE(matrix[1] == 2);
    EXPECT_TRUE(matrix[2] == 3);
    EXPECT_TRUE(matrix[3] == 4);
    EXPECT_TRUE(matrix[4] == 5);
    EXPECT_TRUE(matrix[5] == 6);

    matrix[4] = 50;
    EXPECT_TRUE(matrix(1, 1) == 50);

    const Matrix<int>& view = matrix;
    EXPECT_TRUE(view[4] == 50);
}

TEST(ContainersMatrix, const_access)
{
    const Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    EXPECT_TRUE(matrix(0, 0) == 1);
    EXPECT_TRUE(matrix.at(1, 1) == 4);
    EXPECT_TRUE(matrix.front() == 1);
    EXPECT_TRUE(matrix.back() == 4);
    EXPECT_TRUE(matrix.data() != nullptr);
}

TEST(ContainersMatrix, iteration)
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    const int sum = std::accumulate(matrix.begin(), matrix.end(), 0);

    EXPECT_TRUE(sum == 10);
}

TEST(ContainersMatrix, const_iteration)
{
    const Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    int expected = 1;

    for (auto it = matrix.cbegin(); it != matrix.cend(); ++it) {
        EXPECT_TRUE(*it == expected);
        ++expected;
    }

    EXPECT_TRUE(expected == 5);
}

TEST(ContainersMatrix, reverse_iteration)
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    int expected = 4;

    for (auto it = matrix.rbegin(); it != matrix.rend(); ++it) {
        EXPECT_TRUE(*it == expected);
        --expected;
    }

    EXPECT_TRUE(expected == 0);
}

TEST(ContainersMatrix, fill)
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    matrix.fill(42);

    for (int value : matrix) {
        EXPECT_TRUE(value == 42);
    }

    Matrix<int> empty(0, 0);
    empty.fill(7);
    EXPECT_TRUE(empty.empty());
}

TEST(ContainersMatrix, copy_constructor)
{
    Matrix<int> original{
        {1, 2},
        {3, 4}
    };
    Matrix<int> copy(original);

    EXPECT_TRUE(copy.rows() == 2);
    EXPECT_TRUE(copy.cols() == 2);
    EXPECT_TRUE(copy(0, 0) == 1);
    EXPECT_TRUE(copy(1, 1) == 4);

    copy(0, 0) = 99;

    EXPECT_TRUE(original(0, 0) == 1);
    EXPECT_TRUE(copy(0, 0) == 99);
}

TEST(ContainersMatrix, copy_assignment)
{
    Matrix<int> original{
        {5, 6},
        {7, 8}
    };
    Matrix<int> copy(0, 0);

    copy = original;

    EXPECT_TRUE(copy.rows() == 2);
    EXPECT_TRUE(copy.cols() == 2);
    EXPECT_TRUE(copy(0, 0) == 5);
    EXPECT_TRUE(copy(1, 1) == 8);

    copy = copy;
    EXPECT_TRUE(copy.rows() == 2);
    EXPECT_TRUE(copy.cols() == 2);
    EXPECT_TRUE(copy(0, 0) == 5);
    EXPECT_TRUE(copy(1, 1) == 8);
}

TEST(ContainersMatrix, move_constructor)
{
    Matrix<int> original{
        {1, 2, 3},
        {4, 5, 6}
    };
    Matrix<int> moved(std::move(original));

    EXPECT_TRUE(moved.rows() == 2);
    EXPECT_TRUE(moved.cols() == 3);
    EXPECT_TRUE(moved(0, 0) == 1);
    EXPECT_TRUE(moved(1, 2) == 6);
}

TEST(ContainersMatrix, move_assignment)
{
    Matrix<int> original{
        {9, 10},
        {11, 12}
    };
    Matrix<int> moved(0, 0);

    moved = std::move(original);

    EXPECT_TRUE(moved.rows() == 2);
    EXPECT_TRUE(moved.cols() == 2);
    EXPECT_TRUE(moved(0, 0) == 9);
    EXPECT_TRUE(moved(1, 1) == 12);

    Matrix<int>& same = moved;
    moved = std::move(same);
    EXPECT_TRUE(moved.rows() == 2);
    EXPECT_TRUE(moved.cols() == 2);
    EXPECT_TRUE(moved(0, 0) == 9);
    EXPECT_TRUE(moved(1, 1) == 12);
}

TEST(ContainersMatrix, swap)
{
    Matrix<int> a{
        {1, 2}
    };
    Matrix<int> b{
        {3, 4},
        {5, 6}
    };

    a.swap(b);

    EXPECT_TRUE(a.rows() == 2);
    EXPECT_TRUE(a.cols() == 2);
    EXPECT_TRUE(a(0, 0) == 3);
    EXPECT_TRUE(a(1, 1) == 6);

    EXPECT_TRUE(b.rows() == 1);
    EXPECT_TRUE(b.cols() == 2);
    EXPECT_TRUE(b(0, 0) == 1);
    EXPECT_TRUE(b(0, 1) == 2);
}

TEST(ContainersMatrix, swap_with_empty)
{
    Matrix<int> populated{
        {1, 2},
        {3, 4}
    };
    Matrix<int> empty(0, 0);

    populated.swap(empty);

    EXPECT_TRUE(populated.empty());
    EXPECT_TRUE(empty.rows() == 2);
    EXPECT_TRUE(empty.cols() == 2);
    EXPECT_TRUE(empty(0, 0) == 1);
    EXPECT_TRUE(empty(1, 1) == 4);
    EXPECT_TRUE(empty.strides().rank() == 2);
    EXPECT_TRUE(empty.strides()(0) == 2);
    EXPECT_TRUE(empty.strides()(1) == 1);
}

