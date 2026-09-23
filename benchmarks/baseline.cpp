// Standalone benchmark workload; scripts/benchmark.py builds and records it.
#include <stratax.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using stratax::container::Matrix;
using stratax::container::Vector;
using stratax::core::ArrayView;
using stratax::core::Shape;

namespace {

template<typename T>
void escape(const T& value)
{
#if defined(__GNUC__) || defined(__clang__)
    // The memory clobber keeps producing the result inside the timed interval.
    asm volatile("" : : "g"(&value) : "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)value;
#endif
}

double checksum(double value)
{
    return value;
}

template<typename Array>
double checksum(const Array& array)
{
    double result = 0;

    for (const auto value : array)
        result += value;

    return result;
}

void verify(double actual, double expected)
{
    if (actual != expected)
        throw std::runtime_error("scalar correctness check failed");
}

template<typename Array>
void verify(
    const Array& actual,
    const std::vector<double>& expected,
    const Shape& expected_shape)
{
    if (actual.shape() != expected_shape ||
        actual.size() != expected.size())
    {
        throw std::runtime_error(
            "result shape correctness check failed"
        );
    }

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        if (actual[i] != expected[i])
        {
            throw std::runtime_error(
                "element correctness check failed"
            );
        }
    }
}

template<typename Operation, typename Validator>
void measure(
    const char* name,
    std::size_t rows,
    std::size_t columns,
    std::size_t warmups,
    std::size_t repeats,
    Operation operation,
    Validator validate)
{
    struct Sample
    {
        std::chrono::nanoseconds::rep elapsed_ns;
        double checksum;
    };

    // Perform one full correctness validation before timing.
    validate(operation());

    // Warm up the benchmark before collecting samples.
    for (std::size_t i = 0; i < warmups; ++i)
    {
        auto result = operation();
        escape(result);
    }

    std::vector<Sample> samples;
    samples.reserve(repeats);

    // Collect all timed samples before writing anything to stdout.
    for (std::size_t i = 0; i < repeats; ++i)
    {
        std::atomic_signal_fence(std::memory_order_seq_cst);

        const auto start =
            std::chrono::steady_clock::now();

        auto result = operation();

        escape(result);

        const auto end =
            std::chrono::steady_clock::now();

        const auto nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start
            ).count();

        // Checksum and result destruction remain outside
        // the timed interval.
        samples.push_back({
            nanoseconds,
            checksum(result)
        });
    }

    // Output is delayed until all timed samples are complete.
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        std::cout
            << name << ','
            << rows << ','
            << columns << ','
            << rows * columns << ','
            << i << ','
            << samples[i].elapsed_ns << ','
            << std::setprecision(17)
            << samples[i].checksum
            << '\n';
    }
}

void run(
    std::size_t rows,
    std::size_t columns,
    std::size_t warmups,
    std::size_t repeats)
{
    if (
        rows == 0 ||
        columns == 0 ||
        columns >
            std::numeric_limits<std::size_t>::max() / 2 ||
        rows >
            std::numeric_limits<std::size_t>::max() /
            (2 * columns))
    {
        throw std::invalid_argument(
            "invalid or overflowing benchmark dimensions"
        );
    }

    Matrix<double> input(rows, columns);
    Matrix<double> spaced(rows, 2 * columns);
    Vector<double> row(columns);

    std::vector<double> values(rows * columns);
    std::vector<double> scalar_expected(values.size());
    std::vector<double> broadcast_expected(values.size());
    std::vector<double> axis0_expected(columns, 0);
    std::vector<double> axis1_expected(rows, 0);

    double total = 0;

    for (std::size_t j = 0; j < columns; ++j)
    {
        row[j] =
            static_cast<double>(j % 13) / 4;
    }

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            const auto flat =
                i * columns + j;

            // Binary fractions make the reference exact
            // at the supplied sizes.
            const double value =
                (static_cast<int>(flat % 251) - 125) /
                8.0;

            input(i, j) = value;

            spaced(i, 2 * j) = value;
            spaced(i, 2 * j + 1) = -999;

            values[flat] = value;

            scalar_expected[flat] =
                value + 3.0;

            broadcast_expected[flat] =
                value + row[j];

            axis0_expected[j] += value;
            axis1_expected[i] += value;

            total += value;
        }
    }

    const ArrayView<const double> contiguous(
        input.data(),
        input.shape(),
        input.strides()
    );

    const ArrayView<const double> strided(
        spaced.data(),
        Shape{rows, columns},
        Shape{2 * columns, 2}
    );

    const auto check_values =
        [&](const auto& result)
        {
            verify(
                result,
                values,
                Shape{rows, columns}
            );
        };

    const auto check_total =
        [&](double result)
        {
            verify(result, total);
        };

    const auto check_axis0 =
        [&](const auto& result)
        {
            verify(
                result,
                axis0_expected,
                Shape{columns}
            );
        };

    const auto check_axis1 =
        [&](const auto& result)
        {
            verify(
                result,
                axis1_expected,
                Shape{rows}
            );
        };

    measure(
        "owning_scalar_add",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return input + 3.0;
        },
        [&](const auto& result)
        {
            verify(
                result,
                scalar_expected,
                input.shape()
            );
        }
    );

    measure(
        "owning_row_broadcast_add",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return input + row;
        },
        [&](const auto& result)
        {
            verify(
                result,
                broadcast_expected,
                input.shape()
            );
        }
    );

    measure(
        "owning_iteration",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return checksum(input);
        },
        check_total
    );

    measure(
        "contiguous_view_iteration",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return checksum(contiguous);
        },
        check_total
    );

    measure(
        "strided_view_iteration",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return checksum(strided);
        },
        check_total
    );

    measure(
        "owning_to_tensor_copy",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return stratax::conversion::to_tensor(input);
        },
        check_values
    );

    measure(
        "contiguous_view_materialize",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return stratax::conversion::to_tensor(contiguous);
        },
        check_values
    );

    measure(
        "strided_view_materialize",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return stratax::conversion::to_tensor(strided);
        },
        check_values
    );

    measure(
        "owning_axis0_sum",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return reduction::sum(input, 0);
        },
        check_axis0
    );

    measure(
        "owning_axis1_sum",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return reduction::sum(input, 1);
        },
        check_axis1
    );

    measure(
        "strided_view_axis1_sum",
        rows,
        columns,
        warmups,
        repeats,
        [&]
        {
            return reduction::sum(strided, 1);
        },
        check_axis1
    );
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 5)
        {
            throw std::invalid_argument(
                "usage: baseline ROWS COLUMNS WARMUPS REPEATS"
            );
        }

        std::cout
            << "case,rows,columns,elements,"
               "sample,elapsed_ns,checksum\n";

        run(
            std::stoull(argv[1]),
            std::stoull(argv[2]),
            std::stoull(argv[3]),
            std::stoull(argv[4])
        );
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "benchmark failed: "
            << error.what()
            << '\n';

        return EXIT_FAILURE;
    }
}