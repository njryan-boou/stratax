@page benchmarks Performance Baselines

# Performance baselines

Run the standalone workload from the repository root with Python 3 and a GCC or
Clang C++20 compiler. It builds a temporary executable with `-O3 -DNDEBUG` and
requires no benchmark library or Python extension.

```sh
python scripts/benchmark.py --sizes 64 256 1024 --repeats 9 \
  --output benchmarks/results/local.json --csv benchmarks/results/local.csv
```

Use `--compiler clang++` to select a compiler, `--columns 128` to make the
requested sizes row counts of rectangular matrices, and `--warmups 3` to change
the warmup count. The driver also reads `CXX`. JSON goes to standard output when
`--output` is omitted; CSV is optional.

The workload compares the same logical matrix values through these paths:

| Case | Work measured |
| --- | --- |
| Owning scalar addition | Allocate a result and add a scalar to every element |
| Owning row broadcast addition | Broadcast a vector across matrix rows |
| Owning, contiguous view, and strided view iteration | Traverse and sum logical elements |
| Owning copy and view materialization | Copy into an owning tensor |
| Axis 0 and axis 1 sums | Reduce an owning matrix along each axis |
| Strided view axis 1 sum | Reduce a view selecting every second stored column |

Inputs and independent expected results are constructed before timing. Every
case checks result shape and all elements, or the scalar total, before warmup.
Each sample times one operation, including its result allocation; checksums,
printing, and destruction of the returned result occur afterward. Iteration cases
include summation as the measured work. The allocation-free iteration cases help
distinguish traversal cost from the allocation and copying in other cases.

Axis reductions currently materialize an initial owning tensor. The separate
`owning_to_tensor_copy` and view-materialization cases give context for that cost;
their timings cannot simply be subtracted from reduction timings because caches,
allocation reuse, and traversal differ. Optimizations should preserve the existing
reduction and view contracts and pass the test suite before comparing timings.

Reports retain raw samples, minimum, median, maximum, dimensions, repeat count,
and checksums. Metadata records the compiler and flags, CPU model, logical CPU
count, OS, Git revision and dirty state, and a hash of the exact headers and
benchmark sources. The hash identifies uncommitted source changes that a revision
alone would miss. JSON and CSV refer to the same samples.

The checked-in `benchmarks/results/baseline-linux-gcc.json` result is a local
reference for this checkout, not a portable performance promise. Run both
revisions on the same idle machine with the same
compiler, flags, sizes, and repeats; compare several independent runs and retain
the raw samples. Inputs are reused across samples without flushing caches;
warmup and untimed checksums influence cache residency, and larger working sets
can exceed cache capacity. CPU frequency, system load, and cache size can dominate
small differences. No timing threshold gates correctness or CI.
