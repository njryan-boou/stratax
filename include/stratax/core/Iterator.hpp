#pragma once

#include <iterator>

namespace stratax::core {

/** @brief Alias template for reverse iterators used by Stratax containers. */
template<typename Iterator>
using reverse_iterator_t = std::reverse_iterator<Iterator>;

} // namespace stratax::core
