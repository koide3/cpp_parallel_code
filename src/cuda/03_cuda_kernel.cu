#include <numeric>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cuda_check_error.hpp>

// GPU上で実行する関数（カーネル）
__global__ void my_kernel(int* inputs, int num_data) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  // i はこの実行スレッドのグローバルID
  printf("%d / %d block=%d thread=%d\n", inputs[i], num_data, blockIdx.x, threadIdx.x);
}

int main(int argc, char** argv) {
  // ストリームを作成する。
  cudaStream_t stream;
  check_error << cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // データを用意する。
  thrust::host_vector<int> h_data(10);
  std::iota(h_data.begin(), h_data.end(), 0);
  thrust::device_vector<int> d_data = h_data;
  int* d_data_ptr = d_data.data().get();

  // ストリーム上でカーネルを起動する（ブロック数=2、スレッド数=5）。
  my_kernel<<<2, 5, 0, stream>>>(d_data_ptr, d_data.size());

  // ストリーム上の処理を待ち、破棄する。
  check_error << cudaStreamSynchronize(stream);
  check_error << cudaStreamDestroy(stream);
}