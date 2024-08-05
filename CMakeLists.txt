cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_ARCHITECTURES 75)
project(cpp_parallel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Eigen is the sole mandatory dependency
find_package(Eigen3)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT Eigen3_FOUND)
  message(STATUS "System Eigen not found. Download Eigen 3.4.0.")
  include(FetchContent)
  FetchContent_Populate(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  )
  add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${eigen3_SOURCE_DIR}"
  )
endif()


file(GLOB STD_SRCS src/std/*.cpp)
foreach(STD_SRC ${STD_SRCS})
  get_filename_component(STD_EXEC ${STD_SRC} NAME_WE)
  add_executable(std_${STD_EXEC} ${STD_SRC})
  target_include_directories(std_${STD_EXEC} PRIVATE
    include
  )
endforeach()


file(GLOB OMP_SRCS src/omp/*.cpp)
foreach(OMP_SRC ${OMP_SRCS})
  get_filename_component(STD_EXEC ${OMP_SRC} NAME_WE)
  add_executable(omp_${STD_EXEC} ${OMP_SRC})
  target_include_directories(omp_${STD_EXEC} PRIVATE
    include
  )
  target_link_libraries(omp_${STD_EXEC} PRIVATE
    OpenMP::OpenMP_CXX
  )
  if(MSVC)
    target_compile_options(omp_${STD_EXEC} PRIVATE /openmp:llvm)
  endif()
endforeach()


file(GLOB TBB_SRCS src/tbb/*.cpp)
foreach(TBB_SRC ${TBB_SRCS})
  get_filename_component(STD_EXEC ${TBB_SRC} NAME_WE)
  add_executable(tbb_${STD_EXEC} ${TBB_SRC})
  target_include_directories(tbb_${STD_EXEC} PRIVATE
    include
  )
  target_link_libraries(tbb_${STD_EXEC} PRIVATE
    TBB::tbb
  )
endforeach()


file(GLOB CUDA_SRCS src/cuda/*.cu)
foreach(CUDA_SRC ${CUDA_SRCS})
  get_filename_component(STD_EXEC ${CUDA_SRC} NAME_WE)
  add_executable(cuda_${STD_EXEC} ${CUDA_SRC})
  target_include_directories(cuda_${STD_EXEC} PRIVATE
    include
  )
  target_link_libraries(cuda_${STD_EXEC} PRIVATE
    Eigen3::Eigen
    CUDA::cudart
  )
  target_compile_options(cuda_${STD_EXEC} PRIVATE
    --expt-extended-lambda
    --expt-relaxed-constexpr
  )
endforeach()

add_executable(cuda_find_transformation
  src/cuda/transformation/find_transformation.cu
  src/cuda/transformation/find_transformation_cpu.cpp
  src/cuda/transformation/find_transformation_gpu_v1.cu
  src/cuda/transformation/find_transformation_gpu_v2.cu
  src/cuda/transformation/find_transformation_gpu_v3.cu
  src/cuda/transformation/find_transformation_gpu_v4.cu
)
target_include_directories(cuda_find_transformation PRIVATE
  include
)
target_link_libraries(cuda_find_transformation PRIVATE
  Eigen3::Eigen
  CUDA::cudart
)
target_compile_options(cuda_find_transformation PRIVATE
  --expt-extended-lambda
  --expt-relaxed-constexpr
)


add_executable(etc_time
  src/etc/time.cu
  src/easy_profiler_cuda.cu
)
target_include_directories(etc_time PRIVATE
  include
)