# GPU Energy Measurement Experiment

A CUDA-based research project for measuring GPU energy consumption during batch matrix multiplication operations using NVIDIA's hardware energy counters. This project investigates the stability and variance of energy measurements across different iteration counts.

## Overview

This experiment measures the energy consumption of GPU operations using NVML (NVIDIA Management Library) hardware counters. It performs batch matrix multiplication operations with varying iteration counts to analyze measurement stability through statistical metrics including mean, standard deviation, and coefficient of variation.

## Features

- **Hardware-level energy measurement** using NVML energy counters
- **Optimized CUDA kernel** with shared memory tiling for batch matrix multiplication
- **Statistical analysis** of energy measurements across multiple trials
- **Configurable experiments** with different iteration counts
- **GPU setup script** for multi-GPU systems to ensure measurement consistency

## Requirements

### Hardware
- NVIDIA GPU with **Volta architecture or newer** (sm_70+)
  - Required for hardware energy counter support
  - Tested architectures: Volta, Turing, Ampere, Ada, Hopper

### Software
- CUDA Toolkit (nvcc compiler)
- NVIDIA drivers with NVML support
- Linux operating system (recommended)
- Root/sudo access (for GPU configuration)

## Quick Start

### Option 1: Using Driver Script (Easiest)

Run the complete workflow with a single command:

```bash
./run_experiment.sh
```

This script automatically:
1. Sets up GPU environment (prompts for sudo if needed)
2. Compiles the program with auto-detected architecture
3. Runs the experiment and saves output to `runlog_GPUMODEL_TIMESTAMP.txt`

### Option 2: Using Makefile

```bash
# Check system requirements
make check

# Build (automatically detects your GPU architecture)
make

# Setup GPU, build, and run everything
make run-full
```

### Option 3: Manual Steps

**1. Configure GPU (Multi-GPU Systems)**

For consistent energy measurements, configure your GPU environment:

```bash
sudo ./setup_gpu.sh
```

This script:
- Enables only GPU 0
- Disables all other GPUs
- Sets persistent mode on GPU 0

**2. Compile**

```bash
nvcc -o batch_matmul_experiment batch_matmul_experiment.cu -lnvml
```

**3. Run**

```bash
./batch_matmul_experiment
```

## Makefile Targets

The Makefile provides convenient commands for building and running:

| Command | Description |
|---------|-------------|
| `make` or `make all` | Build (auto-detects GPU architecture) |
| `make ARCH=sm_80` | Build for specific architecture (overrides auto-detection) |
| `make run` | Build and run the experiment |
| `make run-full` | Setup GPU, build, and run (complete workflow) |
| `make setup` | Configure GPU environment (requires sudo) |
| `make restore-gpu` | Restore GPU to default configuration |
| `make check` | Check system requirements and GPU capabilities |
| `make gpu-status` | Show current GPU configuration |
| `make clean` | Remove build artifacts |
| `make help` | Show all available targets |

**Architecture Override Examples:**

The Makefile automatically detects your GPU architecture. To override:
```bash
make ARCH=sm_70  # Volta (V100)
make ARCH=sm_75  # Turing (RTX 20xx)
make ARCH=sm_80  # Ampere (A100, RTX 30xx)
make ARCH=sm_89  # Ada (RTX 40xx)
make ARCH=sm_90  # Hopper (H100)
```

**Note:** Auto-detection uses `nvidia-smi` to query your GPU's compute capability. If detection fails, it falls back to sm_70.

## Experiment Logs

The `run_experiment.sh` script automatically saves all output to timestamped log files with GPU model tags:

```
runlog_rtx3090_20251206_023045.txt  # Format: runlog_GPUMODEL_YYYYMMDD_HHMMSS.txt
runlog_rtx4090_20251206_103045.txt
runlog_v100sxm232gb_20251206_143045.txt
runlog_a100sxm440gb_20251206_183045.txt
```

**GPU Name Formatting:**
- Spaces and special characters are removed
- Converted to lowercase
- "NVIDIA", "GeForce", "Tesla" prefixes are stripped
- Examples: "RTX 3090" → `rtx3090`, "Tesla V100" → `v100sxm232gb`, "A100-SXM4-40GB" → `a100sxm440gb`

**Log file contents:**
- GPU configuration details
- Compilation output
- Complete experimental results
- Statistical analysis for each iteration count
- Timing and energy measurements

**Viewing logs:**
```bash
# View the most recent log
ls -lt runlog_*.txt | head -n1 | xargs cat

# View logs for specific GPU
cat runlog_rtx3090_*.txt

# Search for specific metrics
grep "Energy per iteration" runlog_*.txt

# Compare results across different GPUs
grep "Energy per iteration" runlog_rtx3090_*.txt runlog_v100_*.txt

# Compare coefficient of variation across runs
grep "CV:" runlog_*.txt
```

## Configuration

Key parameters are defined at the top of `batch_matmul_experiment.cu`:

```c
#define BATCH_SIZE 16        // Number of matrix pairs to process
#define MATRIX_SIZE 4096     // Dimension of square matrices (NxN)
#define TILE_SIZE 32         // Shared memory tile size
```

Experimental parameters in `main()`:

```c
int num_trials = 10;                              // Trials per experiment
int iteration_counts[] = {1, 2, 4, 8, 16};       // Iterations to test
```

## Understanding the Output

The program outputs results for each experiment configuration:

```
=== Experiment: 8 iterations per trial, 10 trials ===
  Trial  1: Total=2.456 J, Per-iter=0.307000 J, Time=145.32 ms
  Trial  2: Total=2.448 J, Per-iter=0.306000 J, Time=145.28 ms
  ...

--- Statistics ---
Energy per iteration: 0.306500 ± 0.000800 J (CV: 0.26%)
Total energy:         2.452 ± 0.006 J
Execution time:       145.30 ± 0.15 ms
Total duration:       145.30 ms (0.15 s)
Range:                [0.305500, 0.307500] J (0.65% variation)
```

### Key Metrics

- **Energy per iteration**: Average energy consumed per kernel execution
- **CV (Coefficient of Variation)**: Measurement stability indicator (lower is better)
- **Range**: Min/max values showing measurement spread
- **Total energy**: Cumulative energy for all iterations in a trial
- **Execution time**: GPU computation time

## Implementation Details

### Energy Monitoring

Uses NVML's `nvmlDeviceGetTotalEnergyConsumption()` API:
- Returns cumulative energy in millijoules
- Hardware-level counter (not estimated)
- Monotonically increasing counter

### Matrix Multiplication Kernel

- **Tiling strategy**: 32x32 shared memory tiles
- **Batch processing**: 3D grid with batch dimension in z-axis
- **Memory optimization**: Coalesced global memory access, reduced latency via shared memory
- **Compute optimization**: Loop unrolling for tile multiplication

### Experimental Design

1. **Warm-up phase**: 5 iterations to stabilize GPU state
2. **Multiple trials**: 10 trials per iteration count for statistical significance
3. **Cooldown periods**: 100ms between trials, 2s between experiments
4. **Iteration counts**: Tests 1, 2, 4, 8, 16 iterations to assess averaging effects

## Troubleshooting

### "Hardware energy counter not supported"

**Solution**: Your GPU architecture is older than Volta (sm_70). Upgrade to a newer GPU.

### "NVML error: Insufficient Permissions"

**Solution**: Run with sudo or ensure proper NVML permissions:
```bash
sudo ./batch_matmul_experiment
```

### High Variance in Measurements

**Possible causes**:
- Other processes using the GPU
- GPU not in persistent mode
- Multiple GPUs interfering

**Solution**: Run `setup_gpu.sh` and close other GPU applications.

### Compilation Errors

**Missing NVML**:
```bash
# Install CUDA toolkit which includes NVML
sudo apt-get install nvidia-cuda-toolkit
```

**Wrong architecture**:
```bash
# Compile for specific GPU architecture (e.g., sm_80 for Ampere)
nvcc -arch=sm_80 -o batch_matmul_experiment batch_matmul_experiment.cu -lnvml
```

## Restoring GPU Configuration

After experiments, restore all GPUs to default state:

```bash
# Reset all GPUs to default compute mode
sudo nvidia-smi -c 0

# Optionally disable persistent mode
sudo nvidia-smi -pm 0
```

Or manually for each GPU:
```bash
sudo nvidia-smi -i 0,1,2,3 -c 0  # Enable all GPUs
```

## File Structure

```
energytime/
├── batch_matmul_experiment.cu   # Main CUDA program
├── run_experiment.sh            # Driver script (setup + compile + run)
├── setup_gpu.sh                 # GPU configuration script
├── Makefile                     # Build automation
├── CLAUDE.md                    # Developer documentation
├── README.md                    # This file
└── runlog_*.txt                 # Experiment output logs (generated)
```

## Research Applications

This tool is useful for:
- Studying GPU energy efficiency
- Benchmarking energy consumption of CUDA kernels
- Investigating measurement variance and stability
- Comparing energy costs of different algorithms
- Reproducible energy profiling research

## License

This is a research project. Check with the repository owner for licensing details.

## Contributing

For bugs or feature requests, please open an issue or submit a pull request.