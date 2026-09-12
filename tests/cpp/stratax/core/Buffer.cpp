#include <gtest/gtest.h>

#include <utility>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <stratax.h>

using namespace stratax::core;

// Constructors

TEST(BufferConstructor, DefaultConstructor)
{
    Buffer<int> buffer;

    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.begin(), buffer.end());
}

TEST(BufferConstructor, SizeConstructor)
{
    Buffer<int> buffer(10);

    EXPECT_EQ(buffer.size(), 10);
    EXPECT_FALSE(buffer.empty());
    EXPECT_NE(buffer.data(), nullptr);

    for (std::size_t i = 0; i < buffer.size(); ++i) 
    {
        EXPECT_EQ(buffer[i], 0);
    }

}

TEST(BufferConstructor, FillConstructor)
{
    Buffer<int> buffer(10, 5);

    EXPECT_EQ(buffer.size(), 10);
    EXPECT_FALSE(buffer.empty());
    EXPECT_NE(buffer.data(), nullptr);

    for (std::size_t i = 0; i < buffer.size(); ++i)
    {
        EXPECT_EQ(buffer[i], 5);
    }
}

TEST(BufferConstructor, ListConstructor)
{
    Buffer<int> buffer{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    EXPECT_EQ(buffer.size(), 10);
    EXPECT_FALSE(buffer.empty());
    EXPECT_NE(buffer.data(), nullptr);

    EXPECT_EQ(buffer[0], 1);
    EXPECT_EQ(buffer[1], 2);
    EXPECT_EQ(buffer[2], 3);
    EXPECT_EQ(buffer[3], 4);
    EXPECT_EQ(buffer[4], 5);
    EXPECT_EQ(buffer[5], 6);
    EXPECT_EQ(buffer[6], 7);
    EXPECT_EQ(buffer[7], 8);
    EXPECT_EQ(buffer[8], 9);
    EXPECT_EQ(buffer[9], 10);
}

TEST(BufferConstructor, ZeroSize)
{
    Buffer<int> sized(0);
    Buffer<int> filled(0, 5);
    Buffer<int> listed(std::initializer_list<int>{});

    EXPECT_TRUE(sized.empty());
    EXPECT_EQ(sized.size(), 0);
    EXPECT_EQ(sized.data(), nullptr);

    EXPECT_TRUE(filled.empty());
    EXPECT_EQ(filled.data(), nullptr);

    EXPECT_TRUE(listed.empty());
    EXPECT_EQ(listed.data(), nullptr);
}

namespace {
struct Tracked
{
    static inline int live = 0;
    static inline int copies_left = -1;
    static inline bool fail_default = false;
    int value = 0;

    Tracked()
    {
        if (fail_default) throw std::runtime_error("default construction failed");
        ++live;
    }
    Tracked(const Tracked& other) : value(other.value)
    {
        if (copies_left == 0) throw std::runtime_error("copy failed");
        if (copies_left > 0) --copies_left;
        ++live;
    }
    Tracked& operator=(const Tracked&) = default;
    ~Tracked() { --live; }
};

class BufferLifetime : public testing::Test
{
    void SetUp() override
    {
        ASSERT_EQ(Tracked::live, 0);
        Tracked::copies_left = -1;
        Tracked::fail_default = false;
    }
    void TearDown() override
    {
        EXPECT_EQ(Tracked::live, 0);
        Tracked::copies_left = -1;
        Tracked::fail_default = false;
    }
};
}

TEST(BufferOwnership, CopiesAreIndependentAndAssignmentReplacesSize)
{
    Buffer<int> source{1, 2, 3};
    Buffer<int> copy(source);
    Buffer<int> assigned{9};
    EXPECT_EQ(&(assigned = source), &assigned);
    EXPECT_NE(copy.data(), source.data());
    EXPECT_NE(assigned.data(), source.data());
    EXPECT_EQ(assigned.size(), 3);
    copy[0] = 8;
    assigned[1] = 9;
    EXPECT_EQ(std::vector<int>(source.begin(), source.end()), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(assigned[2], 3);
}

TEST(BufferOwnership, MovesTransferStorageAndAllowSourceReuse)
{
    Buffer<int> source{1, 2, 3};
    auto* storage = source.data();
    Buffer<int> moved(std::move(source));
    EXPECT_EQ(moved.data(), storage);
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.size(), 0);
    EXPECT_EQ(source.begin(), source.end());
    source = Buffer<int>{7};
    EXPECT_EQ(source.front(), 7);
    EXPECT_EQ(&(source = std::move(moved)), &source);
    EXPECT_EQ(source.data(), storage);
    EXPECT_EQ(source.size(), 3);
    EXPECT_EQ(moved.data(), nullptr);
    EXPECT_EQ(moved.size(), 0);
}

TEST(BufferOwnership, SelfAssignmentAndSwapPreserveStorage)
{
    Buffer<int> source{1, 2, 3};
    auto* storage = source.data();
    auto& alias = source;
    source = alias;
    source = std::move(alias);
    source.swap(source);
    EXPECT_EQ(source.data(), storage);
    EXPECT_EQ(source.size(), 3);
    Buffer<int> empty;
    source.swap(empty);
    EXPECT_EQ(empty.data(), storage);
    EXPECT_EQ(empty.back(), 3);
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_TRUE(source.empty());
    empty = source;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.data(), nullptr);
}

TEST(BufferStorage, AlignmentAndAllocationOverflow)
{
    Buffer<double, 128> values(7);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(values.data()) % 128, 0);
    EXPECT_THROW(Buffer<double>(Buffer<double>::max_size() + 1), std::bad_array_new_length);
}

TEST(BufferStorage, FillAndConstReverseIteration)
{
    Buffer<int> values{1, 2, 3};
    const auto& read_only = values;
    EXPECT_EQ(std::vector<int>(read_only.rbegin(), read_only.rend()), (std::vector<int>{3, 2, 1}));
    values.fill(4);
    EXPECT_EQ(std::vector<int>(read_only.cbegin(), read_only.cend()), (std::vector<int>{4, 4, 4}));
    Buffer<int> empty;
    empty.fill(3);
    EXPECT_EQ(empty.crbegin(), empty.crend());
}

TEST_F(BufferLifetime, MoveDoesNotCopyElementsAndDestroysOldStorage)
{
    Buffer<Tracked> source(3);
    Buffer<Tracked> destination(2);
    Tracked::copies_left = 0;
    auto* storage = source.data();
    destination = std::move(source);
    EXPECT_EQ(destination.data(), storage);
    EXPECT_EQ(Tracked::live, 3);
    Buffer<Tracked> moved(std::move(destination));
    EXPECT_EQ(moved.data(), storage);
    EXPECT_EQ(Tracked::live, 3);
}

TEST_F(BufferLifetime, FailedCopyConstructionDestroysPartialElements)
{
    Buffer<Tracked> source(3);
    for (int failure = 0; failure < 3; ++failure) {
        Tracked::copies_left = failure;
        EXPECT_THROW((Buffer<Tracked>(source)), std::runtime_error);
        EXPECT_EQ(Tracked::live, 3);
    }
}

TEST_F(BufferLifetime, FailedCopyAssignmentLeavesDestinationUnchanged)
{
    Buffer<Tracked> source(3);
    Buffer<Tracked> destination(2);
    destination[0].value = 42;
    auto* storage = destination.data();
    for (int failure = 0; failure < 3; ++failure) {
        Tracked::copies_left = failure;
        EXPECT_THROW(destination = source, std::runtime_error);
        EXPECT_EQ(destination.data(), storage);
        EXPECT_EQ(destination.size(), 2);
        EXPECT_EQ(destination[0].value, 42);
        EXPECT_EQ(Tracked::live, 5);
    }
}

TEST_F(BufferLifetime, FailedFillAndListConstructionCleanUp)
{
    Tracked value;
    const std::initializer_list<Tracked> values{value, value, value};
    for (int failure = 0; failure < 3; ++failure) {
        Tracked::copies_left = failure;
        EXPECT_THROW((Buffer<Tracked>(3, value)), std::runtime_error);
        EXPECT_EQ(Tracked::live, 4);
        Tracked::copies_left = failure;
        EXPECT_THROW((Buffer<Tracked>(values)), std::runtime_error);
        EXPECT_EQ(Tracked::live, 4);
    }
}

TEST_F(BufferLifetime, FailedDefaultConstructionReleasesAllocation)
{
    // LeakSanitizer also checks the raw allocation when construction fails.
    Tracked::fail_default = true;
    EXPECT_THROW(Buffer<Tracked>(3), std::runtime_error);
    EXPECT_EQ(Tracked::live, 0);
}
