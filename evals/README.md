# Reproducing GPU Benchmark Results

This artifact describes how to reproduce the quantitative GPU benchmark results
presented in the paper. Prism generates CUDA kernels from high-level
specifications; this evaluation compiles and benchmarks those kernels on real
hardware (NVIDIA H100 and RTX 4090) against vendor libraries (cuBLAS, CUB).

The eval scripts handle everything end-to-end: provisioning cloud GPUs via
[RunPod](https://www.runpod.io/), uploading generated code, compiling,
benchmarking, collecting results, and tearing down the machines. **No local GPU
is required.** The full evaluation takes less than 30 minutes.

## Prerequisites

- **Python 3** (for kernel generation and plotting)
- **Standard tools:** `curl`, `ssh`, `tar` (pre-installed on macOS and Linux)
- **RunPod account** with an API key and SSH public key configured

If you are a reviewer, you should have received a RunPod API key on HotCRP.
Please also provide your public SSH key on HotCRP so we can add it to the
RunPod account.

```bash
export RUNPOD_API_KEY="YOUR_RUNPOD_KEY"
```

## Reproducing All Results

From the repository root:

```bash
# 1. Generate CUDA kernels from Prism specifications
python generate_evals.py

# 2. Run benchmarks
cd evals
./run_h100.sh           # H100 BF16 GEMM (~5 min)
./run_4090.sh           # RTX 4090 scan + SGEMM (~15 min)

# 3. Plot results
python plot_results.py results/*.txt
# Plots saved to results/plots/
```

The sections below describe each benchmark in more detail and options for
running them individually.

> **Note:** The RTX 4090 can be in high demand on RunPod. If a script reports
> no available GPUs, try again after a few minutes.

## Benchmark Details

### H100 BF16 GEMM [~10 Minutes]

Runs a matrix size sweep (2048 to 16384) comparing the Prism-generated BF16
matmul kernel against cuBLAS on an H100 80GB.

```bash
./run_h100.sh                    # full sweep
./run_h100.sh "2048 4096"        # custom sizes
```

### RTX 4090: Prefix Sum (Scan)  [~10 Minutes]

Runs correctness verification and a size sweep (1M to 1B elements) comparing
the Prism-generated scan kernel against NVIDIA CUB.

```bash
./run_4090.sh scan
```

### RTX 4090: SGEMM [~10 Minutes]

Runs correctness verification and a size sweep (256 to 8192) for five
Prism-generated SGEMM kernels of increasing sophistication, compared against
cuBLAS.

```bash
./run_4090.sh sgemm
```

To run both scan and SGEMM on a single pod:

```bash
./run_4090.sh
```

## Mapping Results to Paper Figures

After running the benchmarks and plotting:

| Paper Figure | Description | Plot File |
|---|---|---|
| Figure 21 | SGEMM performance vs cuBLAS (RTX 4090) | `*_sgemm.png` |
| Figure 22 | Prefix sum bandwidth vs CUB (RTX 4090) | `*_scan.png` |
| Figure 23 | BF16 GEMM performance vs cuBLAS (H100) | `*_h100_gemm.png` |

### A Note on Hardware

The SGEMM benchmarks in the paper were collected on an RTX 4070 Ti Super. Due
to limited availability on RunPod, the evaluation scripts use an RTX 4090
instead (same SM 8.9 architecture). The absolute numbers will differ, but the
relative performance trends remain the same.

## Directory Structure

```
run_h100.sh          # H100 benchmark script
run_4090.sh          # RTX 4090 benchmark script
plot_results.py      # Parse results and generate comparison plots
h100_bench/          # H100 BF16 GEMM harness
4090_scan/           # Prefix sum benchmark harness
4090_sgemm/          # SGEMM sweep benchmark harness
results/             # Benchmark output and plots (gitignored)
```

## How It Works

Each eval script follows the same flow:

1. **Sync** generated kernels from `../generated/` into the benchmark directory
2. **Provision** a GPU pod on RunPod via their GraphQL API
3. **Upload** benchmark source code to the pod via `tar` over SSH
4. **Compile** with `nvcc` on the pod
5. **Run** benchmarks, streaming results back to a local file
6. **Tear down** the pod (guaranteed via `trap` + a remote watchdog timer)

The pod is always cleaned up, even on errors or Ctrl-C.
