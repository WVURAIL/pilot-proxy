# ==============================================================================
# F-Statistic CUDA Kernel Makefile
# ==============================================================================
#
# Build targets:
#   make           - Build shared library (release)
#   make debug     - Build with debug symbols and bounds checking
#   make clean     - Remove build artifacts
#   make info      - Print configuration
#
# Configuration:
#   Override GPU architecture: make SM=86
#   Override compiler:         make NVCC=/usr/local/cuda/bin/nvcc
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Toolchain
# ------------------------------------------------------------------------------
NVCC                  ?= nvcc
CC                    ?= gcc
CXX                   ?= g++
SM                    ?= 89
DP4A                  ?= 1
BLOCK_THREADS         ?= 64
GRID_MAX_BLOCKS       ?= 4096
CONSTANT_WEIGHT_LANES ?= 0
SHARED_WEIGHT_LANES   ?= 1

# ------------------------------------------------------------------------------
# Compiler Flags
# ------------------------------------------------------------------------------
NVCC_BASE   := -std=c++14 -Xcompiler -fPIC --shared

# Architecture flags
ARCH_FLAGS  := -arch=sm_$(SM)

# Warning flags
WARN_FLAGS  := -Xcompiler -Wall

# Configuration defines shared by the library and CUDA regression test.
CONFIG_DEFS := \
	-DFSTAT_USE_DP4A=$(DP4A) \
	-DFSTAT_BLOCK_THREADS=$(BLOCK_THREADS) \
	-DFSTAT_GRID_MAX_BLOCKS=$(GRID_MAX_BLOCKS) \
	-DFSTAT_USE_CONSTANT_WEIGHT_LANES=$(CONSTANT_WEIGHT_LANES) \
	-DFSTAT_USE_SHARED_WEIGHT_LANES=$(SHARED_WEIGHT_LANES)

# Release flags (default)
OPT_FLAGS   := -O3 -DNDEBUG -use_fast_math -lineinfo $(CONFIG_DEFS)

# Debug flags
DEBUG_FLAGS := -O0 -g -G -DDEBUG $(CONFIG_DEFS)

# Combined flags
RELEASE_FLAGS := $(NVCC_BASE) $(ARCH_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS)
DEBUG_BUILD   := $(NVCC_BASE) $(ARCH_FLAGS) $(DEBUG_FLAGS) $(WARN_FLAGS)

# ------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------
override TARGET   := libfstatistic.so
SOURCES           := f_statistic.cu
HEADERS           := f_statistic.h config.h
REF_TEST          := test_fstat_reference
REF_TEST_SOURCES  := f_statistic_reference.cpp tests/test_fstat_reference.cpp
REF_TEST_HEADERS  := f_statistic_reference.h config.h
C_HEADER_TEST_OBJ := test_c_header.o
CUDA_TEST         := test_fstat_cuda
CUDA_TEST_SOURCES := tests/test_fstat_cuda.cpp f_statistic_reference.cpp
CUDA_TEST_HEADERS := f_statistic.h f_statistic_reference.h config.h

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------
.PHONY: all debug clean info test_c_header test_ref test_cuda test_cuda_grid_stride test

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(RELEASE_FLAGS) -o $@ $(SOURCES)
	@echo "Built $(TARGET) for SM $(SM)"

debug: $(SOURCES) $(HEADERS)
	$(NVCC) $(DEBUG_BUILD) -o $(TARGET) $(SOURCES)
	@echo "Built $(TARGET) (DEBUG) for SM $(SM)"

$(REF_TEST): $(REF_TEST_SOURCES) $(REF_TEST_HEADERS)
	$(CXX) -std=c++14 -O2 -Wall -I. -o $@ $(REF_TEST_SOURCES)
	@echo "Built $(REF_TEST)"

$(C_HEADER_TEST_OBJ): tests/test_c_header.c f_statistic.h config.h
	$(CC) -std=c11 -Wall -I. -c tests/test_c_header.c -o $@
	@echo "Compiled C header smoke test"

test_c_header: $(C_HEADER_TEST_OBJ)

test_ref: $(REF_TEST)
	./$(REF_TEST)

$(CUDA_TEST): $(TARGET) $(CUDA_TEST_SOURCES) $(CUDA_TEST_HEADERS)
	$(NVCC) -std=c++14 -O2 -I. $(CONFIG_DEFS) $(CUDA_TEST_SOURCES) \
	    -L. -lfstatistic -Xlinker -rpath -Xlinker . -o $@
	@echo "Built $(CUDA_TEST)"

test_cuda: $(CUDA_TEST)
	./$(CUDA_TEST)

test_cuda_grid_stride:
	$(MAKE) clean
	$(MAKE) test_cuda GRID_MAX_BLOCKS=4

test: test_c_header test_ref test_cuda

clean:
	rm -f $(TARGET) $(REF_TEST) $(CUDA_TEST) $(C_HEADER_TEST_OBJ)
	@echo "Cleaned build artifacts"

info:
	@echo "========================================"
	@echo "F-Statistic Kernel Build Configuration"
	@echo "========================================"
	@echo "Compiler:     $(NVCC)"
	@echo "C compiler:   $(CC)"
	@echo "C++ compiler: $(CXX)"
	@echo "Target:       $(TARGET)"
	@echo "SM Version:   $(SM)"
	@echo "DP4A:         $(DP4A)"
	@echo "Block threads: $(BLOCK_THREADS)"
	@echo "Grid max blocks: $(GRID_MAX_BLOCKS)"
	@echo "Constant weight lanes: $(CONSTANT_WEIGHT_LANES)"
	@echo "Shared weight lanes: $(SHARED_WEIGHT_LANES)"
	@echo "Release:      $(RELEASE_FLAGS)"
	@echo "Debug:        $(DEBUG_BUILD)"
	@echo "========================================"

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
.PHONY: help
help:
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all     Build release library (default)"
	@echo "  debug   Build with debug symbols"
	@echo "  test    Build and run C header, CPU reference, and CUDA regression tests"
	@echo "  test_c_header"
	@echo "          Compile public C header smoke test"
	@echo "  test_ref"
	@echo "          Build and run CPU reference tests"
	@echo "  test_cuda"
	@echo "          Build and run CUDA-vs-CPU reference tests"
	@echo "  test_cuda_grid_stride"
	@echo "          Rebuild with GRID_MAX_BLOCKS=4 and run CUDA regression tests"
	@echo "  clean   Remove build artifacts"
	@echo "  info    Print build configuration"
	@echo "  help    Show this message"
	@echo ""
	@echo "Options:"
	@echo "  SM=XX   Set compute capability (default: 89)"
	@echo "  GRID_MAX_BLOCKS=N"
	@echo "          Set maximum power-accumulation grid blocks (default: 4096)"
	@echo "  CONSTANT_WEIGHT_LANES=0|1"
	@echo "          Store DP4A weight lanes in CUDA constant memory (default: 0)"
	@echo "  SHARED_WEIGHT_LANES=0|1"
	@echo "          Preload DP4A weight lanes into per-block shared memory (default: 1)"
	@echo "          Common values: 70 (V100), 80 (A100), 86 (RTX 30xx),"
	@echo "                         89 (RTX 40xx/Ada), 90 (H100)"
	@echo ""
	@echo "Examples:"
	@echo "  make SM=89          Build for RTX 5000 Ada"
	@echo "  make debug SM=80    Debug build for A100"
