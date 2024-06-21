import ctypes

import numpy as np

# Requires openmp installed via homebrew
# Use .so for suffix rather then .dylib even though we are on macOS
# g++ -shared -fPIC -O3 -flto -fpermissive -I/opt/homebrew/Cellar/libomp/18.1.6/include -std=c++11 -Xpreprocessor -fopenmp -L/opt/homebrew/Cellar/libomp/18.1.6/lib -lomp -o update_ages.so update_ages.cpp

update_ages_lib = ctypes.CDLL("./update_ages.so")
update_ages_lib.update_ages.argtypes = [
    ctypes.c_size_t,  # start_idx
    ctypes.c_size_t,  # stop_idx
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
]

ages = np.random.randint(-4, 28, 1024).astype(np.float32)
bges = np.array(ages)

update_ages_lib.update_ages(0, len(ages), bges)
print(ages[0:16])
print(bges[0:16])
