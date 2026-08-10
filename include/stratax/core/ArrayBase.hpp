#pragma once

namespace stratax::core {

/** @brief Minimal CRTP base used as a common marker for array-like types. */
template<typename Derived>
class ArrayBase
{
protected:
    ~ArrayBase() = default;
};

} // namespace stratax::core
