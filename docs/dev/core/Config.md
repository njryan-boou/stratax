@page dev_config Config

# Config

Header: `include/stratax/core/Config.hpp`.
All constants are in `stratax::core::config`.

| Constant | Value | Current effect |
| --- | --- | --- |
| `default_alignment` | 64 | Default Buffer allocation alignment in bytes |
| `bounds_checking` | true | Reserved; accessors do not consult it |
| `use_simd` | true | Reserved; no operation dispatches on it |
| `use_multithreading` | false | Reserved; no operation dispatches on it |
| `use_cuda` | false | Reserved; no operation dispatches on it |

Bounds behavior is defined by each accessor: at() checks, while owning flat
operator[] and multidimensional operator() require valid indices. The flags do
not enable or disable those checks, nor provide execution backends.
Buffer also accepts an explicit alignment template argument, which must be a
power of two and at least alignof(T). See @ref buffer.
