#!/bin/bash

# Driver script for GPU Energy Measurement Experiment
# This script automates the complete workflow:
# 1. Setup GPU environment
# 2. Compile the program
# 3. Run the experiment with logging

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect GPU name and sanitize for filename
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | \
           sed 's/NVIDIA GeForce //g; s/NVIDIA //g; s/Tesla //g' | \
           tr -d ' ' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')

# Fallback to "unknown" if GPU detection fails
if [ -z "$GPU_NAME" ]; then
    GPU_NAME="unknown"
fi

# Log file name with GPU model and timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="runlog_${GPU_NAME}_${TIMESTAMP}.txt"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GPU Energy Measurement Experiment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Setup GPU
echo -e "${YELLOW}[Step 1/3] Setting up GPU environment...${NC}"
if [ -f "./setup_gpu.sh" ]; then
    if [ "$EUID" -ne 0 ]; then
        echo "GPU setup requires root privileges. You may be prompted for your password."
        sudo ./setup_gpu.sh
    else
        ./setup_gpu.sh
    fi
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ GPU setup complete${NC}"
    else
        echo -e "${RED}✗ GPU setup failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ setup_gpu.sh not found, skipping GPU setup${NC}"
fi
echo ""

# Step 2: Compile
echo -e "${YELLOW}[Step 2/3] Compiling the program...${NC}"
make clean
make
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
else
    echo -e "${RED}✗ Compilation failed${NC}"
    exit 1
fi
echo ""

# Step 3: Run experiment
echo -e "${YELLOW}[Step 3/3] Running experiment...${NC}"
echo -e "${BLUE}Output will be logged to: ${LOGFILE}${NC}"
echo ""

# Check if executable exists
if [ ! -f "./batch_matmul_experiment" ]; then
    echo -e "${RED}✗ Executable not found: ./batch_matmul_experiment${NC}"
    exit 1
fi

# Run the experiment with tee to log output
./batch_matmul_experiment 2>&1 | tee "$LOGFILE"

# Check if experiment completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Experiment completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${BLUE}Log saved to: ${LOGFILE}${NC}"
    echo ""

    # Show log file size and location
    LOG_SIZE=$(du -h "$LOGFILE" | cut -f1)
    echo -e "Log file details:"
    echo -e "  Path: ${SCRIPT_DIR}/${LOGFILE}"
    echo -e "  Size: ${LOG_SIZE}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Experiment failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${YELLOW}Check ${LOGFILE} for details${NC}"
    exit 1
fi
