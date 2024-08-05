#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>

#include <cuda_check_error.hpp>

// 点間の二乗距離を計算するカーネル。v1 と同一。
struct evaluate_error_kernel_v2 {
  __device__ float operator()(const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f>& inputs) const {
    const Eigen::Vector3f target = thrust::get<0>(inputs);
    const Eigen::Vector3f source = thrust::get<1>(inputs);
    return (target - (T.block<3, 3>(0, 0) * source + T.block<3, 1>(0, 3))).squaredNorm();
  }

  Eigen::Matrix4f T;  // 評価する相対姿勢。
};

// 二つの点群間の二乗距離を計算する。
// v1 から引数に処理を実行するストリームが増えている。
float evaluate_error_gpu_v2(
  const thrust::device_vector<Eigen::Vector3f>& target_points,
  const thrust::device_vector<Eigen::Vector3f>& source_points,
  const Eigen::Matrix4f& T,
  cudaStream_t stream) {
  // 計算した二乗距離を格納するGPU上の配列。
  // thrust::device_vector は同期が起きてしまうので、cudaMallocAsyncを使ってメモリ確保する。
  float* errors;
  check_error << cudaMallocAsync(&errors, sizeof(float) * target_points.size(), stream);

  // ターゲットとソース点のペアに対して二乗距離を計算する。
  // 第一引数で処理を実行するストリームを指定している。
  thrust::transform(
    thrust::cuda::par_nosync.on(stream),
    thrust::make_zip_iterator(target_points.begin(), source_points.begin()),
    thrust::make_zip_iterator(target_points.end(), source_points.end()),
    errors,
    evaluate_error_kernel_v2{T});

  // 二乗誤差の総和を求める。
  // これも第一引数で処理を実行するストリームを指定している。
  // なお、thrust::reduce は追加メモリ確保とデータのCPUへの転送が必要なので、ストリーム指定しても同期が起きる。
  float sum_errors = thrust::reduce(thrust::cuda::par_nosync.on(stream), errors, errors + target_points.size());

  // 誤差を格納していた配列を解放する。
  check_error << cudaFreeAsync(errors, stream);

  return sum_errors;
}

// 二つの点群間の誤差を最小化する。
// ストリームが増えた以外は v1 と同一。
Eigen::Isometry3f find_transformation_gpu_v2(const thrust::device_vector<Eigen::Vector3f>& target_points, const thrust::device_vector<Eigen::Vector3f>& source_points) {
  // ストリームを作成する。
  cudaStream_t stream;
  check_error << cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // 乱数生成器と姿勢ノイズ分布
  std::mt19937 mt;
  std::normal_distribution<> noise_dist(0.0, 0.1);

  // 現在のベスト姿勢とベスト誤差（最小誤差）
  Eigen::Matrix4f best_T = Eigen::Matrix4f::Identity();
  double best_error = evaluate_error_gpu_v2(target_points, source_points, best_T, stream);

  for (int i = 0; i < 8192; i++) {
    // 適当な姿勢ノイズを生成する。
    Eigen::Matrix4f noise = Eigen::Matrix4f::Identity();
    noise.block<3, 1>(0, 3) << noise_dist(mt), noise_dist(mt), noise_dist(mt);
    noise.block<3, 3>(0, 0) = Eigen::Quaternionf(1.0f, noise_dist(mt), noise_dist(mt), noise_dist(mt)).normalized().toRotationMatrix();

    // 現在の最適姿勢にノイズを加え、誤差を計算する。
    Eigen::Matrix4f T = best_T * noise;
    double error = evaluate_error_gpu_v2(target_points, source_points, T, stream);

    // もし誤差がベスト値より小さければ、ベスト姿勢とベスト誤差を更新する。
    if (error < best_error) {
      best_error = error;
      best_T = T;
    }
  }

  // ストリームを破棄する。
  check_error << cudaStreamDestroy(stream);

  return Eigen::Isometry3f(best_T);
}
