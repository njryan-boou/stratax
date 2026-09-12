/** @file
 * @brief Internal result staging for compound operations on aliased arrays.
 */
#pragma once

#include <stratax/core/dtypes/Concepts.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace stratax::core::compound_detail {

/**
 * @brief Writes computed values, staging them when storage may overlap.
 *
 * Staging evaluates every logical result before the first write. Repeated
 * destination offsets receive the last result in logical row-major order.
 * Owning Stratax containers have unique storage and may skip staging when
 * their other operand also owns storage or is a copied scalar.
 *
 * @tparam Stage Whether all results must be computed before writing.
 * @param lhs Mutable destination array.
 * @param evaluate Callable returning the result at a logical flat index.
 * @return Reference to @p lhs.
 * @pre The callable does not modify either operand or its metadata.
 * @throws Any exception propagated by allocation, indexing, or @p evaluate.
 * @note With staging, evaluation failure leaves the destination unchanged.
 *       Without staging, a throwing callable may leave earlier writes intact.
 * @complexity O(n) evaluations and writes; O(n) extra element storage if staged.
 * @internal
 */
template<bool Stage, Array A, typename Evaluate>
A& write_results(A& lhs, Evaluate evaluate)
{
	const auto count = lhs.size();
	if constexpr (Stage)
	{
		std::vector<typename A::value_type> results;
		results.reserve(count);
		for (std::size_t i = 0; i < count; ++i)
		{
			results.push_back(static_cast<typename A::value_type>(evaluate(i)));
		}
		for (std::size_t i = 0; i < count; ++i)
		{
			lhs[i] = results[i];
		}
	}
	else
	{
		for (std::size_t i = 0; i < count; ++i)
		{
			lhs[i] = static_cast<typename A::value_type>(evaluate(i));
		}
	}
	return lhs;
}

/** @brief True for a Stratax container with unique owning element storage. @internal */
template<typename A>
inline constexpr bool owns_storage = is_array<std::remove_cvref_t<A>>::value;

} // namespace stratax::core::compound_detail
