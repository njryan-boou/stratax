// TODO: make size() overflow-safe for extreme ptrdiff_t bounds and steps.
#pragma once

#include <cstddef>

#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::core {

/**
 * @brief Describes a signed, strided half-open index range.
 *
 * Slice stores raw `[start, stop)` bounds and a nonzero step. A positive step
 * selects values while they are less than @p stop; a negative step selects
 * values while they are greater than @p stop. Bounds are not normalized or
 * clamped by this class because normalization depends on the dimension being
 * sliced.
 *
 * @par Example
 * @code
 * const stratax::core::Slice forward{1, 8, 2};
 * // Selects 1, 3, 5, 7; forward.size() == 4.
 *
 * const stratax::core::Slice reverse{7, 0, -2};
 * // Selects 7, 5, 3, 1; reverse.size() == 4.
 * @endcode
 *
 * @see stratax::indexing::slice
 */
class Slice
{
public:
	/** @brief Unsigned type used for the number of selected indices. */
	using size_type = std::size_t;
	/** @brief Signed type used for bounds and step values. */
	using difference_type = std::ptrdiff_t;

private:
	difference_type start_;
	difference_type stop_;
	difference_type step_;

public:
	/**
	 * @brief Constructs a slice from raw half-open bounds and a step.
	 *
	 * The values are stored unchanged. In particular, negative bounds are not
	 * converted relative to a container extent during construction.
	 *
	 * @param start First index in the range when the range is non-empty.
	 * @param stop Exclusive boundary that terminates the range.
	 * @param step Nonzero increment between consecutive indices.
	 * @throws Exceptions::IndexError If @p step is zero.
	 * @complexity O(1).
	 */
	Slice(difference_type start, difference_type stop, difference_type step = 1)
		: start_(start),
		  stop_(stop),
		  step_(step)
	{
		if (step == 0) {
			throw Exceptions::IndexError("Slice step cannot be zero.");
		}
	}

	/** @brief Returns the raw inclusive start bound. @complexity O(1). */
	[[nodiscard]] difference_type start() const noexcept {return start_;}
	/** @brief Returns the raw exclusive stop bound. @complexity O(1). */
	[[nodiscard]] difference_type stop() const noexcept {return stop_;}
	/** @brief Returns the nonzero signed step. @complexity O(1). */
	[[nodiscard]] difference_type step() const noexcept {return step_;}

	/**
	 * @brief Returns the number of indices described by the raw range.
	 *
	 * A positive-step slice is empty when `start() >= stop()`. A negative-step
	 * slice is empty when `start() <= stop()`. Otherwise the result is the
	 * ceiling of the directed distance divided by the absolute step.
	 *
	 * This calculation does not normalize bounds against a container extent.
	 *
	 * @return Number of generated indices.
	 * @pre Intermediate signed distance and rounding arithmetic must be
	 *      representable by difference_type.
	 * @complexity O(1).
	 */
	[[nodiscard]] size_type size() const noexcept
	{
		if (step_ > 0)
		{
			if (start_ >= stop_)
			{
				return 0;
			}

			const difference_type distance = stop_ - start_;
			return static_cast<size_type>((distance + step_ - 1) / step_);
		}

		if (start_ <= stop_)
		{
			return 0;
		}

		const difference_type stride = -step_;
		const difference_type distance = start_ - stop_;
		return static_cast<size_type>((distance + stride - 1) / stride);
	}

	/** @brief Reports whether the raw range selects no indices. @complexity O(1). */
	[[nodiscard]] bool empty() const noexcept {return size() == 0;}

	/**
	 * @brief Compares two slices by their stored bounds and step.
	 * @param other Slice to compare with this slice.
	 * @return `true` when start, stop, and step are all equal.
	 * @complexity O(1).
	 */
	[[nodiscard]] bool operator==(const Slice& other) const noexcept {return start_ == other.start_ && stop_ == other.stop_ && step_ == other.step_;}
};

}
