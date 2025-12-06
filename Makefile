# Makefile for GPU Energy Measurement Experiment

# Compiler and flags
NVCC := nvcc
CUDA_FLAGS := -O3

# Auto-detect GPU architecture, fallback to sm_70 if detection fails
DETECTED_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.' | sed 's/^/sm_/' 2>/dev/null)
ARCH ?= $(if $(DETECTED_ARCH),$(DETECTED_ARCH),sm_70)

# Auto-detect CUDA path
CUDA_PATH ?= $(shell dirname $$(dirname $$(which nvcc 2>/dev/null) 2>/dev/null) 2>/dev/null)

# Library paths and flags
NVML_LIB_PATH := $(CUDA_PATH)/targets/x86_64-linux/lib/stubs
LDFLAGS := -L$(NVML_LIB_PATH)
LIBS := -lnvidia-ml

# Target executable
TARGET := batch_matmul_experiment

# Source files
SOURCES := batch_matmul_experiment.cu

# Setup script
SETUP_SCRIPT := setup_gpu.sh

# Default target
.PHONY: all
all: $(TARGET)

# Compile the CUDA program
$(TARGET): $(SOURCES)
	@echo "Compiling $(TARGET) for architecture $(ARCH)..."
	@if [ "$(DETECTED_ARCH)" != "" ]; then \
		echo "  (auto-detected from GPU)"; \
	fi
	$(NVCC) $(CUDA_FLAGS) -arch=$(ARCH) $(LDFLAGS) -o $(TARGET) $(SOURCES) $(LIBS)
	@echo "Build complete: $(TARGET)"

# Setup GPU environment (requires sudo)
.PHONY: setup
setup:
	@echo "Configuring GPU environment..."
	@if [ ! -x "$(SETUP_SCRIPT)" ]; then \
		chmod +x $(SETUP_SCRIPT); \
	fi
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Note: GPU setup requires root privileges"; \
		sudo ./$(SETUP_SCRIPT); \
	else \
		./$(SETUP_SCRIPT); \
	fi

# Run the experiment
.PHONY: run
run: $(TARGET)
	@echo "Running experiment..."
	./$(TARGET)

# Run with GPU setup first
.PHONY: run-full
run-full: setup $(TARGET)
	@echo "Running experiment with configured GPU..."
	./$(TARGET)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET)
	@echo "Clean complete"

# Deep clean (including any output files if they exist)
.PHONY: distclean
distclean: clean
	@echo "Deep clean complete"

# Restore GPU configuration to default
.PHONY: restore-gpu
restore-gpu:
	@echo "Restoring GPU configuration to default..."
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Note: Restoring GPU requires root privileges"; \
		sudo nvidia-smi -c 0; \
		sudo nvidia-smi -pm 0; \
	else \
		nvidia-smi -c 0; \
		nvidia-smi -pm 0; \
	fi
	@echo "GPU configuration restored"

# Check system requirements
.PHONY: check
check:
	@echo "Checking system requirements..."
	@echo -n "CUDA compiler (nvcc): "; \
	if command -v nvcc >/dev/null 2>&1; then \
		echo "✓ Found ($$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//'))"; \
	else \
		echo "✗ Not found"; \
	fi
	@echo -n "NVIDIA driver (nvidia-smi): "; \
	if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "✓ Found (Driver version: $$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1))"; \
	else \
		echo "✗ Not found"; \
	fi
	@echo -n "GPU detected: "; \
	if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi --query-gpu=name --format=csv,noheader | head -n1; \
		echo -n "  Compute capability: "; \
		nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1; \
	else \
		echo "No GPU detected"; \
	fi
	@echo -n "Energy counter support: "; \
	if nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | awk '{if ($$1 >= 7.0) exit 0; else exit 1}' 2>/dev/null; then \
		echo "✓ Supported (Volta or newer)"; \
	else \
		echo "✗ Not supported (requires Volta/sm_70 or newer)"; \
	fi

# Show GPU status
.PHONY: gpu-status
gpu-status:
	@echo "Current GPU configuration:"
	@nvidia-smi --query-gpu=index,name,persistence_mode,compute_mode,power.draw,temperature.gpu --format=csv

# Help target
.PHONY: help
help:
	@echo "GPU Energy Measurement Experiment - Makefile targets:"
	@echo ""
	@echo "Build targets:"
	@echo "  make           - Build the program (auto-detects GPU architecture: $(ARCH))"
	@echo "  make ARCH=sm_80 - Build for specific architecture (e.g., sm_80 for Ampere)"
	@echo ""
	@echo "Run targets:"
	@echo "  make run       - Build and run the experiment"
	@echo "  make run-full  - Setup GPU, build, and run the experiment"
	@echo ""
	@echo "Setup targets:"
	@echo "  make setup     - Configure GPU environment (requires sudo)"
	@echo "  make restore-gpu - Restore GPU to default configuration (requires sudo)"
	@echo ""
	@echo "Utility targets:"
	@echo "  make check     - Check system requirements"
	@echo "  make gpu-status - Show current GPU configuration"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make distclean - Deep clean"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Common architectures:"
	@echo "  sm_70  - Volta (Tesla V100)"
	@echo "  sm_75  - Turing (RTX 20xx, GTX 16xx)"
	@echo "  sm_80  - Ampere (A100, RTX 30xx)"
	@echo "  sm_86  - Ampere (RTX 30xx mobile)"
	@echo "  sm_89  - Ada Lovelace (RTX 40xx)"
	@echo "  sm_90  - Hopper (H100)"
