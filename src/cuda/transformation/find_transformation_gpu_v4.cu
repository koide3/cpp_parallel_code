#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <cuda_check_error.hpp>

// 点間の二乗距離を計算するカーネル。
// ★ best_T と noise から評価すべき姿勢(T = best_T * noise)を求めるようになっている。
struct evaluate_error_kernel_v4 {
  __device__ float operator()(int i) const {
    const Eigen::Vector3f target = target_points[i];
    const Eigen::Vector3f source = source_points[i];
    const Eigen::Matrix4f T = (*best_T) * (*noise);  // ★

    return (target - (T.block<3, 3>(0, 0) * source + T.block<3, 1>(0, 3))).squaredNorm();
  }

  const Eigen::Vector3f* target_points;
  const Eigen::Vector3f* source_points;
  const Eigen::Matrix4f* best_T;
  const Eigen::Matrix4f* noise;
};

// ★ GPU上の変数を初期化するカーネル。
__global__ void initialize_kernel_v4(Eigen::Matrix4f* d_best_T, float* d_best_error) {
  *d_best_T = Eigen::Matrix4f::Identity();            // ★ 現在までのベスト姿勢
  *d_best_error = std::numeric_limits<float>::max();  // ★ 現在までのベスト誤差
}

// ★ ベスト値を更新するカーネル。
__global__ void update_best_kernel_v4(Eigen::Matrix4f* d_best_T, const Eigen::Matrix4f* d_noise, float* d_best_error, const float* d_current_error) {
  // ★ 現在の誤差値がベスト値より小さければ、ベスト値を更新する。
  if (*d_current_error < *d_best_error) {
    *d_best_error = *d_current_error;
    *d_best_T = (*d_best_T) * (*d_noise);
  }
}

// ★ 二つの点群間の二乗距離を計算し、ベスト値を更新する。
void evaluate_and_update_error_v4(
  const thrust::device_vector<Eigen::Vector3f>& target_points,
  const thrust::device_vector<Eigen::Vector3f>& source_points,
  Eigen::Matrix4f* d_best_T,
  const Eigen::Matrix4f* d_noise,
  float* d_best_error,
  float* d_current_error,
  void** d_temp_storage,
  size_t& d_temp_storage_bytes,
  cudaStream_t stream) {
  // ★ 距離計算カーネルと入力イテレータを作成する。
  evaluate_error_kernel_v4 kernel{target_points.data().get(), source_points.data().get(), d_best_T, d_noise};
  cub::TransformInputIterator<float, evaluate_error_kernel_v4, thrust::counting_iterator<int>> begin(thrust::make_counting_iterator<int>(0), kernel);

  // DeviceReduce::Sum に必要な計算バッファサイズを計算する。
  size_t required_temp_storage_bytes = 0;
  check_error << cub::DeviceReduce::Sum(static_cast<void*>(nullptr), required_temp_storage_bytes, begin, d_current_error, target_points.size(), stream);

  // 必要な計算バッファサイズが、今のバッファサイズより大きければ新しくバッファを確保しなおす。
  if (required_temp_storage_bytes > d_temp_storage_bytes) {
    check_error << cudaFreeAsync(*d_temp_storage, stream);
    check_error << cudaMallocAsync(d_temp_storage, required_temp_storage_bytes, stream);
    d_temp_storage_bytes = required_temp_storage_bytes;
  }

  // DeviceReduce::Sum を実行する。
  check_error << cub::DeviceReduce::Sum(*d_temp_storage, d_temp_storage_bytes, begin, d_current_error, target_points.size(), stream);

  // ★ 評価した誤差値がベスト値より小さければ、それでベスト値を更新する。
  // ★ この更新もGPU上の指定したストリーム上で行われるので、CPUとの同期はここでは必要ない。
  update_best_kernel_v4<<<1, 1, 0, stream>>>(d_best_T, d_noise, d_best_error, d_current_error);
}

// 二つの点群間の誤差を最小化する。
Eigen::Isometry3f find_transformation_gpu_v4(const thrust::device_vector<Eigen::Vector3f>& target_points, const thrust::device_vector<Eigen::Vector3f>& source_points) {
  // ストリームを作成する。
  cudaStream_t stream;
  check_error << cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // 乱数生成器と姿勢ノイズ分布
  std::mt19937 mt;
  std::normal_distribution<> noise_dist(0.0, 0.1);

  // ★ 処理を一気にGPU上で実行するため、使用する姿勢ノイズを事前に生成しておいてまとめてアップロードしておく。
  std::vector<Eigen::Matrix4f> h_noises(8193);
  h_noises[0].setIdentity();
  for (int i = 1; i < h_noises.size(); i++) {
    h_noises[i].setIdentity();
    h_noises[i].block<3, 1>(0, 3) << noise_dist(mt), noise_dist(mt), noise_dist(mt);
    h_noises[i].block<3, 3>(0, 0) = Eigen::Quaternionf(1.0f, noise_dist(mt), noise_dist(mt), noise_dist(mt)).normalized().toRotationMatrix();
  }

  // ★ 姿勢ノイズをGPU上にアップロード。
  Eigen::Matrix4f* d_noises;
  check_error << cudaMallocAsync(&d_noises, sizeof(Eigen::Matrix4f) * h_noises.size(), stream);
  check_error << cudaMemcpyAsync(d_noises, h_noises.data(), sizeof(Eigen::Matrix4f) * h_noises.size(), cudaMemcpyHostToDevice, stream);

  // ★ ベスト姿勢とベスト誤差、現在の誤差評価結果を格納する領域を確保する（全てGPU上）。
  Eigen::Matrix4f* d_best_T;
  float* d_best_error;
  float* d_current_error;
  check_error << cudaMallocAsync(&d_best_T, sizeof(Eigen::Matrix4f), stream);
  check_error << cudaMallocAsync(&d_best_error, sizeof(float), stream);
  check_error << cudaMallocAsync(&d_current_error, sizeof(float), stream);

  // ★ ベスト値を初期化する。
  initialize_kernel_v4<<<1, 1, 0, stream>>>(d_best_T, d_best_error);

  // reduce の計算バッファ。
  void* d_temp_storage = nullptr;
  size_t d_temp_storage_bytes = 0;

  // ★ 生成しておいた姿勢ノイズへのポインタを引数として誤差計算とベスト値更新を実行していく。
  for (int i = 0; i < h_noises.size(); i++) {
    evaluate_and_update_error_v4(target_points, source_points, d_best_T, d_noises + i, d_best_error, d_current_error, &d_temp_storage, d_temp_storage_bytes, stream);
  }

  // ★ 更新後のベスト姿勢をダウンロードする。
  // ★ ここでは通常のCPUメモリなので自動的に同期が行われるが、念のためストリーム同期を呼んでおく。
  Eigen::Matrix4f h_best_T;
  check_error << cudaMemcpyAsync(&h_best_T, d_best_T, sizeof(Eigen::Matrix4f), cudaMemcpyDeviceToHost, stream);
  check_error << cudaStreamSynchronize(stream);

  // ★ ストリーム・メモリの破棄
  check_error << cudaFreeAsync(d_best_T, stream);
  check_error << cudaFreeAsync(d_best_error, stream);
  check_error << cudaFreeAsync(d_current_error, stream);

  check_error << cudaFreeAsync(d_noises, stream);
  check_error << cudaFreeAsync(d_temp_storage, stream);

  check_error << cudaStreamDestroy(stream);

  return Eigen::Isometry3f(h_best_T);
}
