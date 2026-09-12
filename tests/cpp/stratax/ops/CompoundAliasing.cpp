#include <gtest/gtest.h>
#include <stratax.h>

#include <array>
#include <functional>
#include <stdexcept>
#include <vector>

using stratax::container::Matrix;
using stratax::container::Vector;
using stratax::core::ArrayView;
using stratax::core::Shape;

namespace {

enum class Operation { Add, Subtract, Multiply, Divide, And, Or, Xor, Left, Right };

int result(Operation operation, int lhs, int rhs)
{
    switch (operation)
    {
    case Operation::Add: return lhs + rhs;
    case Operation::Subtract: return lhs - rhs;
    case Operation::Multiply: return lhs * rhs;
    case Operation::Divide: return lhs / rhs;
    case Operation::And: return lhs & rhs;
    case Operation::Or: return lhs | rhs;
    case Operation::Xor: return lhs ^ rhs;
    case Operation::Left: return lhs << rhs;
    case Operation::Right: return lhs >> rhs;
    }
    throw std::logic_error("Unexpected operation");
}

template<class L, class R>
void assign(Operation operation, L& lhs, const R& rhs)
{
    switch (operation)
    {
    case Operation::Add: lhs += rhs; break;
    case Operation::Subtract: lhs -= rhs; break;
    case Operation::Multiply: lhs *= rhs; break;
    case Operation::Divide: lhs /= rhs; break;
    case Operation::And: lhs &= rhs; break;
    case Operation::Or: lhs |= rhs; break;
    case Operation::Xor: lhs ^= rhs; break;
    case Operation::Left: lhs <<= rhs; break;
    case Operation::Right: lhs >>= rhs; break;
    }
}

constexpr auto operations = std::array{
    Operation::Add, Operation::Subtract, Operation::Multiply, Operation::Divide,
    Operation::And, Operation::Or, Operation::Xor, Operation::Left, Operation::Right};

} // namespace

TEST(CompoundAliasing, GeneratedOverlapsMatchIndependentOperandSnapshots)
{
    // Physical offsets are generated independently of ArrayView's indexing.
    // Include overlap in either direction, interleaved elements, disjoint
    // elements, exact self-aliasing, and repeated destination/source offsets.
    for (const auto operation : operations)
    for (std::size_t count = 0; count <= 5; ++count)
    for (std::size_t left_offset = 0; left_offset <= 3; ++left_offset)
    for (std::size_t right_offset = 0; right_offset <= 3; ++right_offset)
    for (std::size_t left_stride = 0; left_stride <= 3; ++left_stride)
    for (std::size_t right_stride = 0; right_stride <= 3; ++right_stride)
    {
        SCOPED_TRACE(::testing::Message() << static_cast<int>(operation) << ' '
            << count << ' ' << left_offset << ' ' << right_offset << ' '
            << left_stride << ' ' << right_stride);
        std::vector<int> storage(20);
        for (std::size_t i = 0; i < storage.size(); ++i)
            storage[i] = 1 + static_cast<int>(i % 4);
        const auto before = storage;
        auto expected = storage;
        for (std::size_t i = 0; i < count; ++i)
        {
            expected[left_offset + i * left_stride] = result(operation,
                before[left_offset + i * left_stride],
                before[right_offset + i * right_stride]);
        }
        ArrayView<int> lhs(storage.data() + left_offset, Shape{count}, {left_stride});
        const ArrayView<const int> rhs(storage.data() + right_offset, Shape{count}, {right_stride});
        assign(operation, lhs, rhs);
        EXPECT_EQ(storage, expected);
    }
}

TEST(CompoundAliasing, BroadcastRowAliasingOwningDestinationUsesInitialValues)
{
    for (const auto operation : operations)
    {
        Matrix<int> lhs{{1, 2, 3}, {4, 1, 2}};
        const std::vector<int> before(lhs.begin(), lhs.end());
        const ArrayView<const int> row(lhs.data(), Shape{1, 3}, {3, 1});
        auto* const data = lhs.data();
        assign(operation, lhs, row);
        EXPECT_EQ(lhs.data(), data);
        EXPECT_EQ(lhs.shape(), Shape({2, 3}));
        for (std::size_t i = 0; i < before.size(); ++i)
            EXPECT_EQ(lhs[i], result(operation, before[i], before[i % 3]));
    }
}

TEST(CompoundAliasing, TransposedSourceUsesInitialValues)
{
    for (const auto operation : operations)
    {
        Matrix<int> lhs{{1, 2}, {3, 4}};
        const ArrayView<const int> transposed(lhs.data(), Shape{2, 2}, {1, 2});
        assign(operation, lhs, transposed);
        EXPECT_EQ(lhs[0], result(operation, 1, 1));
        EXPECT_EQ(lhs[1], result(operation, 2, 3));
        EXPECT_EQ(lhs[2], result(operation, 3, 2));
        EXPECT_EQ(lhs[3], result(operation, 4, 4));
    }
}

TEST(CompoundAliasing, AliasedScalarIsCopiedForOwningAndViewDestinations)
{
    for (const auto operation : operations)
    for (std::size_t scalar_index = 0; scalar_index < 4; ++scalar_index)
    {
        Vector<int> owning{1, 2, 3, 4};
        const std::vector<int> before(owning.begin(), owning.end());
        assign(operation, owning, owning[scalar_index]);
        for (std::size_t i = 0; i < before.size(); ++i)
            EXPECT_EQ(owning[i], result(operation, before[i], before[scalar_index]));

        auto storage = before;
        ArrayView<int> view(storage.data(), Shape{2, 2}, {1, 1});
        auto expected = before;
        // Repeated middle offset has the same initial left/scalar values.
        for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            expected[i + j] = result(operation, before[i + j], before[scalar_index]);
        assign(operation, view, storage[scalar_index]);
        EXPECT_EQ(storage, expected);
    }
}

TEST(CompoundAliasing, OverlappingNonzeroStridesUseLastLogicalWrite)
{
    std::array<int, 3> storage{10, 20, 30};
    ArrayView<int> lhs(storage.data(), Shape{2, 2}, {1, 1});
    Matrix<int> rhs{{1, 2}, {3, 4}};
    lhs += rhs;
    EXPECT_EQ(storage, (std::array<int, 3>{11, 23, 34}));
}

TEST(CompoundAliasing, OwningSelfAssignmentPreservesSnapshotResults)
{
    for (const auto operation : operations)
    {
        Vector<int> lhs{1, 2, 3, 4};
        assign(operation, lhs, lhs);
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(lhs[i], result(operation, i + 1, i + 1));
    }
}

TEST(CompoundAliasing, InvalidDivisorOrShiftLeavesWholeAllocationUnchanged)
{
    for (const auto operation : {Operation::Divide, Operation::Left, Operation::Right})
    for (const int invalid : {0, -1, 32})
    {
        if (operation == Operation::Divide && invalid != 0) continue;
        if (operation != Operation::Divide && invalid == 0) continue;
        std::vector<int> storage{2, 3, invalid, 4, 5};
        const auto before = storage;
        ArrayView<int> lhs(storage.data() + 1, Shape{3}, {1});
        const ArrayView<const int> rhs(storage.data(), Shape{3}, {1});
        if (operation == Operation::Divide)
            EXPECT_THROW(assign(operation, lhs, rhs), Exceptions::ZeroDivisionError);
        else
            EXPECT_THROW(assign(operation, lhs, rhs), Exceptions::ValueError);
        EXPECT_EQ(storage, before);
    }
}

TEST(CompoundAliasing, ShapeExpansionAndThrowingCallableDoNotWriteStagedResults)
{
    std::array<int, 4> storage{1, 2, 3, 4};
    const auto before = storage;
    ArrayView<int> lhs(storage.data(), Shape{2}, {1});
    const ArrayView<const int> rhs(storage.data(), Shape{2, 2}, {2, 1});
    for (const auto operation : operations)
    {
        EXPECT_THROW(assign(operation, lhs, rhs), Exceptions::BroadcastError);
        EXPECT_EQ(storage, before);
    }
    int calls = 0;
    const auto throwing = [&calls](int left, int right) {
        if (++calls == 2) throw std::runtime_error("second element");
        return left + right;
    };
    EXPECT_THROW(compound_scalar_op(lhs, 1, throwing), std::runtime_error);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(storage, before);
}

TEST(CompoundAliasing, EmptyViewsNeverAccessStorage)
{
    ArrayView<int> lhs(nullptr, Shape{}, Shape{});
    const ArrayView<const int> rhs(nullptr, Shape{}, Shape{});
    for (const auto operation : operations)
    {
        EXPECT_NO_THROW(assign(operation, lhs, rhs));
        EXPECT_NO_THROW(assign(operation, lhs, 1));
        EXPECT_EQ(lhs.size(), 0U);
    }
    EXPECT_THROW(lhs /= 0, Exceptions::ZeroDivisionError);
    EXPECT_THROW(lhs <<= -1, Exceptions::ValueError);
}
