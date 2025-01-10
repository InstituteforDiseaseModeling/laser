# Compiler and flags
#CXX = g++
CXX = g++-14
CXXFLAGS = -shared -O3 -march=native -fPIC -flto -fopenmp -ffast-math
#CXXFLAGS = -shared -g -march=native -fPIC -flto -fopenmp -ffast-math

# Directories
SRC_DIR = .
BUILD_DIR = ../src/idmlaser_cholera/mods

# Find all .c files in the source directory
SOURCES = $(wildcard $(SRC_DIR)/*.c)

# Generate corresponding .so filenames in the build directory
LIBS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/lib%.so)

# Default target: build all shared libraries
all: $(LIBS)

# Rule to build .so from .c
$(BUILD_DIR)/lib%.so: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean target to remove compiled libraries
clean:
	rm -f $(BUILD_DIR)/lib*.so

