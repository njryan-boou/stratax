# ClassName

Version: vX.Y.Z

Status: Draft

Header: include/stratax/path/ClassName.hpp

---

## Overview

Brief description of what this class provides.

---

## Responsibilities

This class is responsible for:

- Responsibility 1
- Responsibility 2
- Responsibility 3

This class is NOT responsible for:

- Non-goal 1
- Non-goal 2

---

## Relationships

```text
ClassName
|- member_1
`- member_2
```

Depends on:

- Dependency 1
- Dependency 2

Used by:

- Consumer 1
- Consumer 2

Related classes:

- RelatedClass1
- RelatedClass2

---

## Internal Data

| Member | Description |
| ------- | ----------- |
| member_1 | Purpose |
| member_2 | Purpose |

Remove this section if there is no internal state.

---

## Invariants

- Invariant 1
- Invariant 2
- Invariant 3

---

## Public Interface

### Constructors

```cpp
ClassName();
ClassName(...);
```

Describe behavior, complexity, and throws.

### Assignment Operators

```cpp
ClassName& operator=(const ClassName&) = default;
ClassName& operator=(ClassName&&) noexcept = default;
```

Describe behavior and complexity.

### Methods

```cpp
std::size_t size() const noexcept;
bool empty() const noexcept;
```

Describe behavior, complexity, and throws where relevant.

### Operators

```cpp
const ValueType& operator[](std::size_t index) const noexcept;
bool operator==(const ClassName& other) const noexcept;
```

Describe behavior and complexity.

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Construction | O(...) |
| Copy/Move | O(...) |
| Access | O(...) |
| Iteration | O(...) |

---

## Examples

```cpp
// Minimal API-accurate usage
```

---

## Design Notes

Key implementation decisions and tradeoffs.

---

## Future Improvements

- Improvement 1
- Improvement 2

---

## See Also

- RelatedClass1
- RelatedClass2