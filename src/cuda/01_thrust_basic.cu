#include <iostream>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cuda_check_error.hpp>

// 数値を二乗するカーネル（関数）
struct square_kernel {
  // __host__ : CPU上で実行できるカーネルを生成する。
  // __device__ : GPU上で実行できるカーネルを生成する。
  __host__ __device__ int operator()(int x) const {  //
    return x * x;
  }
};

int main(int argc, char** argv) {
  // CPU上で入力配列を用意する。
  thrust::host_vector<int> h_inputs(5);
  for (int i = 0; i < h_inputs.size(); i++) {
    h_inputs[i] = i;
  }

  // GPU上に入力配列をコピー。出力配列も確保する。
  thrust::device_vector<int> d_inputs = h_inputs;
  thrust::device_vector<int> d_outputs(d_inputs.size());

  // GPU上で入力配列の各要素にカーネルを適用し、結果を出力配列に保存する。
  square_kernel kernel;
  thrust::transform(thrust::cuda::par, d_inputs.begin(), d_inputs.end(), d_outputs.begin(), kernel);

  // ラムダ式版
  // thrust::transform(thrust::cuda::par, d_inputs.begin(), d_inputs.end(), d_outputs.begin(), [] __device__(int i) { return i * i; });

  // GPU上の計算結果をCPU上にコピーする。
  thrust::host_vector<int> h_outputs = d_outputs;

  for (int i = 0; i < h_outputs.size(); i++) {
    std::cout << i << ":" << h_outputs[i] << std::endl;
  }

  return 0;
}