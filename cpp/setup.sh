#!/bin/bash
# NeuroDriver C++ Inference — Build Setup
#
# This script handles LibTorch discovery and building on Apple Silicon.
#
# Usage:
#   cd cpp
#   chmod +x setup.sh
#   ./setup.sh

set -e

echo "================================================="
echo "  NeuroDriver C++ Build Setup"
echo "================================================="
echo ""

# --- Step 1: Find LibTorch ---
# Priority: PyTorch's cmake path > Homebrew > Manual

TORCH_CMAKE=""

# Try 1: Get cmake path from installed PyTorch (most reliable on M3)
if command -v python3 &>/dev/null; then
    TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || true)
    if [ -n "$TORCH_CMAKE" ] && [ -d "$TORCH_CMAKE" ]; then
        echo "Found LibTorch via pip/conda PyTorch:"
        echo "  $TORCH_CMAKE"
    else
        TORCH_CMAKE=""
    fi
fi

# Try 2: Homebrew
if [ -z "$TORCH_CMAKE" ]; then
    BREW_TORCH=$(brew --prefix pytorch 2>/dev/null || true)
    if [ -n "$BREW_TORCH" ] && [ -d "$BREW_TORCH/share/cmake/Torch" ]; then
        TORCH_CMAKE="$BREW_TORCH/share/cmake/Torch"
        echo "Found LibTorch via Homebrew:"
        echo "  $TORCH_CMAKE"
    fi
fi

if [ -z "$TORCH_CMAKE" ]; then
    echo "ERROR: LibTorch not found."
    echo ""
    echo "Install via one of:"
    echo "  1. pip install torch          (then re-run this script)"
    echo "  2. brew install pytorch       (then re-run this script)"
    echo "  3. Set CMAKE_PREFIX_PATH manually:"
    echo "     cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .."
    exit 1
fi

# --- Step 2: Export model if not done ---
if [ ! -f "models/driving_model.pt" ]; then
    echo ""
    echo "Model not exported yet. Running export..."
    cd ..
    python3 scripts/export_model.py --output-dir cpp/models
    cd cpp
    echo ""
fi

# --- Step 3: Build ---
echo ""
echo "Building..."
mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" ..
make -j$(sysctl -n hw.ncpu)

echo ""
echo "================================================="
echo "  Build complete!"
echo "================================================="
echo ""
echo "Run inference:"
echo "  ./build/neurodriver_inference models/driving_model.pt ../data_raw/transfuser/<route>"
echo ""
echo "Or single image:"
echo "  ./build/neurodriver_inference models/driving_model.pt ../data_raw/transfuser/<route>/rgb/0000.jpg"