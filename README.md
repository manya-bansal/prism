# Prism

Prism is a GPU programming language that restores modularity without sacrificing low-level control over collective operations. Its central abstraction is **typed perspectives**, which encode at the type level the granularity at which a programmer reasons about and controls thread behavior. By making perspectives explicit, Prism ensures that operations execute with an appropriate view of the underlying hardware, unlocking compositional GPU programming without compromising performance.

This README walks you through setting up the project, running the tests, and reproducing the evaluation from the paper.

---

# Stage 1: Getting Started

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Project Structure

The repo is organized around the core compiler in `prism/`, example programs in `examples/`, and performance benchmarks in `evals/`:

```
prism/            # Compiler and core language implementation
  perspective.py  # Typed perspective abstractions
  type_checker.py # Type checking for perspective safety
  codegen.py      # CUDA code generation
  decorator.py    # Prism kernel decorator
  types.py        # Core type definitions

examples/         # Example Prism programs (SGEMM kernels, scans, MMA, etc.)
generated/        # Generated CUDA output
test/             # Test suite
evals/            # Performance benchmarks and evaluation scripts
```

## Testing

Run the full test suite with:

```bash
pytest
```

## Generating CUDA from Examples

Compile the example Prism programs into CUDA and write them to `generated/`:

```bash
python generate_evals.py
```

## Getting Access to Machines

If you are an artifact evaluation reviewer, please provide your public SSH key on HotCRP. You should have also received a RunPod API key on HotCRP.

---

# Stage 2: Reproducing the Evaluation

The evaluation provisions cloud GPUs (H100 and RTX 4090) via RunPod, compiles
and benchmarks the generated kernels against vendor libraries (cuBLAS, CUB), and
produces plots corresponding to Figures 21-23 in the paper. No local GPU is
required. The full evaluation takes less than 30 minutes.

See [evals/README.md](evals/README.md) for detailed instructions.

---

# Reusing this Artifact 

There are two main ways to reuse and extend this artifact.

## Writing Programs in Prism

The `examples/` directory contains complete Prism programs ranging from simple
shared-memory SGEMM to H100 WGMMA matmul with TMA. These serve as the best
starting point for writing your own kernels.

A Prism kernel is a Python function decorated with `@prism("global")`. You
express parallelism through **typed perspectives** — annotations like
`grid[1]`, `block[1]`, `thread[1]` — that tell the compiler at what
granularity each variable is accessed. The compiler uses these to automatically
insert synchronization, manage shared memory layout, and generate correct CUDA.

To write a new kernel:

1. Define your kernel function in a new file under `examples/`
2. Annotate parameters and variables with perspectives
3. Use `partition` to carve data across thread hierarchy levels and `group` to
   scope shared state
4. Generate CUDA with `generate("your_kernel_name", auto_sync=True)`

### Your First Prism Program

Here is a complete kernel that adds a constant `m` to every element of an array:

```python
from prism import (
    prism, grid, thread,
    requires, group, partition,
    ptr, const, generate,
)

@requires(grid[1], thread[1])
@prism("global")
def add_m_thread(x: ptr(const(int)) @ grid[1],
                 y: ptr(int) @ grid[1],
                 m: int @ grid[1]):
    tid: int @ thread[1] = id()
    with partition(x, p=thread[1], offset=tid) as x_thread:
        with partition(y, p=thread[1], offset=tid) as y_thread:
            with group(thread[1]):
                y_thread[0] = x_thread[0] + m

if __name__ == "__main__":
    print(generate("add_m_thread"))
```

Let's break this down:

- **`@requires(grid[1], thread[1])`** declares that this kernel uses a 1D grid
  and 1D threads — no blocks or warps needed for this simple case.
- **`@ grid[1]`** on the parameters means `x`, `y`, and `m` are visible to the
  entire grid (they are global arguments).
- **`tid: int @ thread[1] = id()`** gets the thread's global index. The
  `@ thread[1]` annotation means each thread has its own value of `tid`.
- **`partition(x, p=thread[1], offset=tid)`** carves out a per-thread view of
  the array — thread `tid` sees element `x[tid]`.
- **`group(thread[1])`** scopes the body to execute at thread granularity.

Running `python examples/add_m_thread.py` produces the CUDA:

```cuda
__global__ void add_m_thread(const int* x, int* y, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    y[tid] = x[tid] + m;
}
```

From here, see `examples/sgemm_shared_mem_block.py` for a kernel that uses
shared memory and block-level perspectives, or `examples/sgemm_kernel_5.py`
for warp-level tiling.

## Contributing to the Compiler

The compiler lives in `prism/` and is organized into a few key modules:

- **`perspective.py`** — Defines the perspective type system (`grid`, `block`,
  `thread`, `warp`) and the rules for how perspectives compose
- **`type_checker.py`** — Validates that perspective annotations are consistent
  and that data is accessed at the correct granularity
- **`codegen.py`** — Translates the typed Prism AST into CUDA C++
- **`sync_analysis.py`** / **`sync_transform.py`** — The auto-sync pass: analyzes
  shared memory access patterns and inserts `__syncthreads()` barriers where
  needed
- **`decorator.py`** — The `@prism` decorator that ties everything together