#include <thread>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <stopwatch.hpp>
#include <easy_profiler.hpp>
#include <easy_profiler_cuda.hpp>
#include <cuda_check_error.hpp>

void stop_watch() {
  std::cout << "*** stopwatch ***" << std::endl;

  // ストップウォッチを作成。
  Stopwatch sw;

  // 経過時間計測。
  sw.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  sw.stop();

  std::cout << sw.msec() << "msec" << std::endl;

  // ラップタイム計測。
  sw.start();
  for (int i = 0; i < 3; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    sw.lap();

    std::cout << i << " : " << sw.msec() << "msec" << std::endl;
  }
}

void easy_profiler() {
  std::cout << "*** easy_profiler ***" << std::endl;

  // 簡易プロファイラを作成。
  EasyProfiler prof("prof");

  // プロファイラに処理名をプッシュし、処理を実行する。
  prof.push("step_0");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  for (int i = 1; i < 4; i++) {
    prof.push("step_" + std::to_string(i));
    std::this_thread::sleep_for(std::chrono::milliseconds(10 * i));
  }
}

void easy_profiler2() {
  std::cout << "*** easy_profiler2 ***" << std::endl;

  // 簡易プロファイラを作成。
  bool enabled = true;
  bool debug = true;
  std::ofstream ofs("prof.txt");

  EasyProfiler prof("prof", enabled, debug, ofs);

  // プロファイラに処理名をプッシュし、処理を実行する。
  prof.push("step_0");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  for (int i = 1; i < 4; i++) {
    prof.push("step_" + std::to_string(i));
    std::this_thread::sleep_for(std::chrono::milliseconds(10 * i));
  }
}

void easy_profiler_cuda() {
  std::cout << "*** easy_profiler_cuda ***" << std::endl;

  // 最初のCUDA APIコールは時間がかかるので、時間計測に影響しないように適当に同期を先に呼んでおく。
  check_error << cudaDeviceSynchronize();
  check_error << cudaDeviceSynchronize();
  check_error << cudaDeviceSynchronize();

  // 処理を行うストリーム。
  cudaStream_t stream;
  check_error << cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // ストリーム上の処理時間を計測するCUDAプロファイラを作成する。
  EasyProfilerCuda prof("prof_cuda", stream);

  // ストリーム上で適当な処理を行う。
  prof.push("create vector", stream);
  thrust::device_vector<int> data(8192);

  prof.push("fill", stream);
  thrust::fill(thrust::cuda::par_nosync.on(stream), data.begin(), data.end(), 1);

  prof.push("transform", stream);
  thrust::transform(thrust::cuda::par_nosync.on(stream), data.begin(), data.end(), data.begin(), thrust::square<int>());

  prof.push("reduce", stream);
  int sum = thrust::reduce(thrust::cuda::par_nosync.on(stream), data.begin(), data.end(), 0);

  prof.push("sync", stream);
  check_error << cudaStreamSynchronize(stream);

  std::cout << "sum=" << sum << std::endl;

  check_error << cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
  stop_watch();
  easy_profiler();
  easy_profiler2();
  easy_profiler_cuda();

  return 0;
}