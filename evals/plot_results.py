#!/usr/bin/env python3
"""Plot benchmark results from eval runs.

Usage:
    python evals/plot_results.py evals/results/4090_*.txt evals/results/h100_*.txt
    python evals/plot_results.py evals/results/4090_20260316_230613.txt  # single file
"""

import re
import sys
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_h100(lines):
    """Parse H100 matmul results.
    Returns list of (size, kernel_tflops, cublas_tflops)."""
    results = []
    kernel_tflops = None
    for line in lines:
        m = re.search(r'Kernel:\s+.*performance:\s+\(\s*([\d.]+)\)\s+TFLOPS.*size:\s+\((\d+)\)', line)
        if m:
            kernel_tflops = float(m.group(1))
            size = int(m.group(2))
        m = re.search(r'cuBLAS:\s+.*performance:\s+\(\s*([\d.]+)\)\s+TFLOPS.*size:\s+\((\d+)\)', line)
        if m and kernel_tflops is not None:
            cublas_tflops = float(m.group(1))
            size = int(m.group(2))
            results.append((size, kernel_tflops, cublas_tflops))
            kernel_tflops = None
    return results


def parse_sgemm(lines):
    """Parse SGEMM sweep results.
    Returns dict: kernel_name -> list of (size, gflops)."""
    results = defaultdict(list)
    current_size = None
    current_kernel = None
    in_perf = False

    for line in lines:
        if 'Starting Performance Sweep' in line:
            in_perf = True
            continue
        if not in_perf:
            continue

        m = re.match(r'Size:\s+(\d+)x(\d+)x(\d+)', line.strip())
        if m:
            current_size = int(m.group(1))
            continue

        if current_size is None:
            continue

        m = re.match(r'\[cuBLAS FP32\]', line.strip())
        if m:
            current_kernel = 'cuBLAS'
            continue

        m = re.match(r'\[My Kernel (\d+)\]', line.strip())
        if m:
            num = int(m.group(1))
            names = {1: 'Naive', 2: 'Mem Coalescing', 3: '2D SMEM Tiling',
                     4: '2D Tiling + Vectorize', 5: 'Warp Tiling'}
            current_kernel = names.get(num, f'Kernel {num}')
            continue

        m = re.search(r'performance:\s+\(\s*([\d.]+)\)\s+GFLOPS.*size:\s+\((\d+)\)', line)
        if m and current_kernel:
            gflops = float(m.group(1))
            results[current_kernel].append((current_size, gflops))
            current_kernel = None

    return dict(results)


def parse_scan(lines):
    """Parse scan vs CUB sweep results.
    Returns list of (size, cub_bw, ours_bw)."""
    results = []
    in_table = False
    for line in lines:
        if 'Size' in line and 'CUB (GB/s)' in line:
            in_table = True
            continue
        if in_table and line.strip().startswith('---'):
            continue
        if in_table and line.strip():
            parts = re.split(r'\s*\|\s*', line.strip())
            if len(parts) >= 3:
                try:
                    size = int(parts[0].strip())
                    cub_parts = parts[1].split()
                    ours_parts = parts[2].split()
                    if len(cub_parts) >= 2 and len(ours_parts) >= 2:
                        cub_bw = float(cub_parts[1])
                        ours_bw = float(ours_parts[1])
                        results.append((size, cub_bw, ours_bw))
                except (ValueError, IndexError):
                    continue
        elif in_table and not line.strip():
            in_table = False
    return results


def plot_h100(results, output_path):
    """Plot H100 matmul: TFLOPS vs size, Prism vs cuBLAS."""
    if not results:
        return
    sizes = [r[0] for r in results]
    kernel = [r[1] for r in results]
    cublas = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, cublas, 'r^-', label='cuBLAS', markersize=8, linewidth=2)
    ax.plot(sizes, kernel, 'bo-', label='Prism', markersize=8, linewidth=2)
    ax.set_xlabel('M = N = K', fontsize=12)
    ax.set_ylabel('TFLOPS/s', fontsize=12)
    ax.set_title('H100 BF16 GEMM Performance', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f'  Saved: {output_path}')
    plt.close(fig)


def plot_sgemm(results, output_path):
    """Plot SGEMM sweep: TFLOPS vs size, multiple kernels."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot order and styles
    kernel_styles = {
        'cuBLAS':               ('r', '^', '-'),
        'Warp Tiling':          ('b', 'o', '-'),
        '2D Tiling + Vectorize':('g', 'p', '-'),
        '2D SMEM Tiling':       ('orange', 'D', '-'),
        'Mem Coalescing':       ('purple', 's', '-'),
        'Naive':                ('goldenrod', '*', '-'),
    }

    for name in ['cuBLAS', 'Warp Tiling', '2D Tiling + Vectorize',
                 '2D SMEM Tiling', 'Mem Coalescing', 'Naive']:
        if name not in results:
            continue
        data = sorted(results[name])
        sizes = [d[0] for d in data]
        gflops = [d[1] / 1000.0 for d in data]  # GFLOPS -> TFLOPS
        color, marker, ls = kernel_styles.get(name, ('gray', 'x', '-'))
        ax.plot(sizes, gflops, color=color, marker=marker, linestyle=ls,
                label=name, markersize=7, linewidth=2)

    ax.set_xlabel('M = N = K', fontsize=12)
    ax.set_ylabel('TFLOPS/s', fontsize=12)
    ax.set_title('SGEMM Performance on RTX 4090', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f'  Saved: {output_path}')
    plt.close(fig)


def plot_scan(results, output_path):
    """Plot scan: Bandwidth vs size, CUB vs Prism."""
    if not results:
        return
    sizes = [r[0] for r in results]
    cub = [r[1] for r in results]
    ours = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, cub, 'r^-', label='CUB', markersize=8, linewidth=2)
    ax.plot(sizes, ours, 'bo-', label='Prism', markersize=8, linewidth=2)
    ax.set_xlabel('Size of Array', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Prefix Sum Performance on RTX 4090', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2E}'))
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f'  Saved: {output_path}')
    plt.close(fig)


def process_file(filepath):
    """Parse a results file and generate plots."""
    with open(filepath) as f:
        lines = f.readlines()

    basename = os.path.splitext(os.path.basename(filepath))[0]
    outdir = os.path.join(os.path.dirname(filepath), 'plots')
    os.makedirs(outdir, exist_ok=True)

    content = ''.join(lines)
    plots_made = 0

    # H100 matmul
    h100_data = parse_h100(lines)
    if h100_data:
        plot_h100(h100_data, os.path.join(outdir, f'{basename}_h100_gemm.png'))
        plots_made += 1

    # SGEMM sweep
    sgemm_data = parse_sgemm(lines)
    if sgemm_data:
        plot_sgemm(sgemm_data, os.path.join(outdir, f'{basename}_sgemm.png'))
        plots_made += 1

    # Scan vs CUB
    scan_data = parse_scan(lines)
    if scan_data:
        plot_scan(scan_data, os.path.join(outdir, f'{basename}_scan.png'))
        plots_made += 1

    if plots_made == 0:
        print(f'  No plottable data found in {filepath}')


def main():
    if len(sys.argv) < 2:
        print('Usage: python evals/plot_results.py <results_file> [results_file ...]')
        sys.exit(1)

    for filepath in sys.argv[1:]:
        print(f'Processing: {filepath}')
        process_file(filepath)


if __name__ == '__main__':
    main()
