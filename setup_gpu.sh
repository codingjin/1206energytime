#!/bin/bash

# GPU Setup Script for Energy Measurement Experiments
# This script:
# 1. Enables only GPU device 0 (disables all other GPUs)
# 2. Sets persistent mode on device 0

set -e  # Exit on error

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    exit 1
fi

echo "========================================="
echo "GPU Configuration for Energy Measurement"
echo "========================================="
echo ""

# Get the number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPU(s)"
echo ""

# List all GPUs
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,persistence_mode --format=csv
echo ""

# Enable persistent mode on GPU 0
echo "Setting persistent mode on GPU 0..."
nvidia-smi -i 0 -pm 1
if [ $? -eq 0 ]; then
    echo "✓ Persistent mode enabled on GPU 0"
else
    echo "✗ Failed to set persistent mode on GPU 0"
    exit 1
fi
echo ""

# Disable all other GPUs if there are multiple GPUs
if [ $NUM_GPUS -gt 1 ]; then
    echo "Disabling GPUs 1-$((NUM_GPUS-1))..."
    for i in $(seq 1 $((NUM_GPUS-1))); do
        echo "  Disabling GPU $i..."
        # Disable compute mode - set to PROHIBITED (3)
        nvidia-smi -i $i -c 3
        if [ $? -eq 0 ]; then
            echo "  ✓ GPU $i disabled"
        else
            echo "  ✗ Failed to disable GPU $i"
        fi
    done
    echo ""
else
    echo "Only one GPU detected, no need to disable others."
    echo ""
fi

# Set GPU 0 to default compute mode (allows all processes)
echo "Setting GPU 0 to default compute mode..."
nvidia-smi -i 0 -c 0
if [ $? -eq 0 ]; then
    echo "✓ GPU 0 set to default compute mode"
else
    echo "✗ Failed to set compute mode on GPU 0"
fi
echo ""

# Display final configuration
echo "========================================="
echo "Final GPU Configuration:"
echo "========================================="
nvidia-smi --query-gpu=index,name,persistence_mode,compute_mode --format=csv
echo ""

echo "Setup complete! GPU 0 is ready for energy measurement experiments."
echo ""
echo "To restore all GPUs, run: sudo nvidia-smi -c 0 -i 0,1,2,..."
