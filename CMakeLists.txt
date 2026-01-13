cmake_minimum_required(VERSION 3.10)
project(CudaHalfPrecisionProject CUDA CXX)

# Set executable name
set(EXEC run)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add CNPY lib
add_library(cnpy
    lib/cnpy/cnpy.cpp
    )
target_include_directories(cnpy PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/cnpy
)
# cnpy depends on zlib
find_package(ZLIB REQUIRED)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)


# Set C++ compiler version
set(cmake_cxx_compiler "/usr/bin/g++-9")

# Find CUDA package
find_package(CUDA 12.1 REQUIRED)
find_package(Eigen3 REQUIRED)
# Attention: Update this using 'which nvcc' (if required)
# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.5/bin/nvcc)
# set(CUDA_TOOLKIT_ROOT_DIR /sw/pkgs/arc/cuda/12.1.1/bin/nvcc)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
# ATTENTION: Update the architecture based on the hardwarde
# Base compilation
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90")
# Disabled optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 -O0 -Xptxas -O0 -Xcompiler -O0 --fmad=false")


# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include/cpp)
include_directories(${CMAKE_SOURCE_DIR}/include/cuda)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# Gather source files
file(GLOB SOURCES "src/cpp/*.cpp" "src/cuda/*.cu")

# Add executable named 'run'
add_executable(${EXEC} ${SOURCES})

# Link CUDA libraries
target_link_libraries(${EXEC} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} cnpy)

# Set CUDA flags
set_target_properties(${EXEC} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CUDA include path
target_include_directories(${EXEC} PRIVATE ${CUDA_INCLUDE_DIRS})
