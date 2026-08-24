#include <gtest/gtest.h>
#include <stratax/exceptions/Exceptions.hpp>

TEST(ExceptionsExceptions, HeaderIsAvailable) { EXPECT_TRUE(true); }

TEST(StrataxError, ExposesStableBaseMetadata)
{
    const Exceptions::StrataxError generic("generic");
    const Exceptions::ShapeError shape("shape");
    const Exceptions::DimensionError dimension("dimension");
    const Exceptions::IndexError index("index");
    const Exceptions::TypeError type("type");
    const Exceptions::BroadcastError broadcast("broadcast");
    const Exceptions::ZeroDivisionError division("division");
    const Exceptions::AxisError axis("axis");
    const Exceptions::OverflowError overflow("overflow");
    const Exceptions::ValueError value("value");

    EXPECT_EQ(generic.code(), Exceptions::ErrorCode::Stratax);
    EXPECT_EQ(shape.code(), Exceptions::ErrorCode::Shape);
    EXPECT_EQ(dimension.code(), Exceptions::ErrorCode::Dimension);
    EXPECT_EQ(index.code(), Exceptions::ErrorCode::Index);
    EXPECT_EQ(type.code(), Exceptions::ErrorCode::Type);
    EXPECT_EQ(broadcast.code(), Exceptions::ErrorCode::Broadcast);
    EXPECT_EQ(division.code(), Exceptions::ErrorCode::ZeroDivision);
    EXPECT_EQ(axis.code(), Exceptions::ErrorCode::Axis);
    EXPECT_EQ(overflow.code(), Exceptions::ErrorCode::Overflow);
    EXPECT_EQ(value.code(), Exceptions::ErrorCode::Value);
    EXPECT_EQ(overflow.category(), "overflow");
}

TEST(StrataxError, MetadataSurvivesBaseClassCatch)
{
    try
    {
        throw Exceptions::AxisError::out_of_range(-3, 2);
    }
    catch (const Exceptions::StrataxError& error)
    {
        EXPECT_EQ(error.code(), Exceptions::ErrorCode::Axis);
        EXPECT_EQ(error.category(), "axis");
        EXPECT_STREQ(error.what(), "Axis -3 is out of range for an array of rank 2. Valid axes range from -2 through 1.");
    }
}

TEST(StrataxError, DirectConstructionCapturesCallSite)
{
    const auto expected_line = __LINE__ + 1;
    const Exceptions::IndexError error(3, 2);

    EXPECT_EQ(error.line(), expected_line);
    EXPECT_NE(error.file_name().find("Exceptions.cpp"), std::string_view::npos);
    EXPECT_FALSE(error.function_name().empty());
    EXPECT_GT(error.column(), 0);
}

TEST(StrataxError, NamedFactoryCapturesCallerInsteadOfFactoryBody)
{
    const auto expected_line = __LINE__ + 1;
    const auto error = Exceptions::IndexError::slice_step_zero();

    EXPECT_EQ(error.line(), expected_line);
    EXPECT_NE(error.file_name().find("Exceptions.cpp"), std::string_view::npos);
}

TEST(IndexError, MessageOnlyConstructionHasNoMetadata)
{
    const Exceptions::IndexError error("custom message");

    EXPECT_STREQ(error.what(), "custom message");
    EXPECT_FALSE(error.has_index_metadata());
    EXPECT_FALSE(error.index().has_value());
    EXPECT_FALSE(error.size().has_value());
}

TEST(IndexError, BoundsConstructionStoresMetadata)
{
    const Exceptions::IndexError error(-4, 3);

    EXPECT_TRUE(error.has_index_metadata());
    ASSERT_TRUE(error.index().has_value());
    ASSERT_TRUE(error.size().has_value());
    EXPECT_EQ(*error.index(), -4);
    EXPECT_EQ(*error.size(), 3);
    EXPECT_STREQ(
        error.what(),
        "Index -4 is out of bounds for size 3. "
        "Valid indices range from -3 through 2.");
}

TEST(IndexError, EmptyBoundsMessageDescribesEmptyArray)
{
    const Exceptions::IndexError error(0, 0);

    EXPECT_STREQ(
        error.what(),
        "Index 0 is out of bounds for size 0. The array is empty.");
}

TEST(IndexError, CustomMessageCanRetainMetadata)
{
    const Exceptions::IndexError error("component error", 3, 3);

    EXPECT_STREQ(error.what(), "component error");
    EXPECT_TRUE(error.has_index_metadata());
    EXPECT_EQ(error.index(), 3);
    EXPECT_EQ(error.size(), 3);
}

TEST(ShapeError, StoresActualAndExpectedShapes)
{
    const Exceptions::ShapeError error(
        Exceptions::ShapeError::shape_type{2, 3},
        Exceptions::ShapeError::shape_type{3, 2});

    EXPECT_TRUE(error.has_shape_metadata());
    EXPECT_EQ(error.actual_shape(), (Exceptions::ShapeError::shape_type{2, 3}));
    EXPECT_EQ(error.expected_shape(), (Exceptions::ShapeError::shape_type{3, 2}));
    EXPECT_STREQ(error.what(), "Shape (2, 3) does not match expected shape (3, 2).");
}

TEST(DimensionError, StoresActualAndExpectedDimensions)
{
    const Exceptions::DimensionError error(3, 2);

    EXPECT_TRUE(error.has_dimension_metadata());
    EXPECT_EQ(error.actual_dimension(), 3);
    EXPECT_EQ(error.expected_dimension(), 2);
}

TEST(TypeError, StoresActualAndExpectedTypes)
{
    const Exceptions::TypeError error("str", "numeric");

    EXPECT_TRUE(error.has_type_metadata());
    EXPECT_EQ(error.actual_type(), "str");
    EXPECT_EQ(error.expected_type(), "numeric");
}

TEST(BroadcastError, StoresBothOperandShapes)
{
    const Exceptions::BroadcastError error({2, 3}, {4, 3});

    EXPECT_TRUE(error.has_shape_metadata());
    EXPECT_EQ(error.left_shape(), (Exceptions::BroadcastError::shape_type{2, 3}));
    EXPECT_EQ(error.right_shape(), (Exceptions::BroadcastError::shape_type{4, 3}));
}

TEST(ZeroDivisionError, StoresDivisorIndexWhenAvailable)
{
    const Exceptions::ZeroDivisionError error(4);

    EXPECT_TRUE(error.has_index_metadata());
    EXPECT_EQ(error.index(), 4);
}

TEST(AxisError, StoresAxisAndRank)
{
    const Exceptions::AxisError error(-4, 3);

    EXPECT_TRUE(error.has_axis_metadata());
    EXPECT_EQ(error.axis(), -4);
    EXPECT_EQ(error.rank(), 3);
    EXPECT_STREQ(error.what(), "Axis -4 is out of range for rank 3.");
}

TEST(OverflowError, StoresOperationAndOperands)
{
    const Exceptions::OverflowError error("multiplication", 8, 9);

    EXPECT_TRUE(error.has_operand_metadata());
    EXPECT_EQ(error.operation(), "multiplication");
    EXPECT_EQ(error.left_operand(), 8);
    EXPECT_EQ(error.right_operand(), 9);
}

TEST(ExceptionFactories, OwnMessagesAndPreserveMetadata)
{
    const auto index = Exceptions::IndexError::multi_index_rank(
        2,
        3,
        Exceptions::IndexError::Context::Tensor);
    const auto value = Exceptions::ValueError::shift_count();

    EXPECT_STREQ(
        index.what(),
        "Tensor multi-index has 2 components, but the target has rank 3; provide exactly one index or slice component per dimension.");
    EXPECT_EQ(index.index(), 2);
    EXPECT_EQ(index.size(), 3);
    EXPECT_STREQ(
        value.what(),
        "Shift count must be non-negative and less than the bit width of the shifted value.");
}
