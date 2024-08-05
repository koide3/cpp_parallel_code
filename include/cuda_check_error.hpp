#pragma once

#include <string>
#include <iostream>

#include <cuda_runtime.h>

class CUDACheckError {
public:
  void operator<<(cudaError_t error) const {
    if (error == cudaSuccess) {
      return;
    }

    const std::string error_name = cudaGetErrorName(error);
    const std::string error_string = cudaGetErrorString(error);

    std::cerr << "warning: " << error_name << std::endl;
    std::cerr << "       : " << error_string << std::endl;
  }
};

static CUDACheckError check_error;