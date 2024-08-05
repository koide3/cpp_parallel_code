#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>

// 点間の二乗距離を計算するカーネル。
struct evaluate_error_kernel_v1 {
  __device__ float operator()(const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f>& inputs) const {
    const Eigen::Vector3f target = thrust::get<0>(inputs);
    const Eigen::Vector3f source = thrust::get<1>(inputs);
    return (target - (T.block<3, 3>(0, 0) * source + T.block<3, 1>(0, 3))).squaredNorm();
  }

  // 評価する相対姿勢。
  // Isometry3f はCUDA上で問題を起こしがちなので、普通の4x4行列を使う。
  Eigen::Matrix4f T;
};

// 二つの点群間の二乗距離を計算する。
float evaluate_error_gpu_v1(const thrust::device_vector<Eigen::Vector3f>& target_points, const thrust::device_vector<Eigen::Vector3f>& source_points, const Eigen::Matrix4f& T) {
  // 計算した二乗距離を格納する配列をGPU上に確保する。
  thrust::device_vector<float> errors(target_points.size());

  // ターゲットとソース点のペアに対して二乗距離を計算する。
  thrust::transform(
    thrust::cuda::par,
    thrust::make_zip_iterator(target_points.begin(), source_points.begin()),
    thrust::make_zip_iterator(target_points.end(), source_points.end()),
    errors.begin(),
    evaluate_error_kernel_v1{T});

  // 二乗誤差の総和を求める。
  return thrust::reduce(thrust::cuda::par, errors.begin(), errors.end());
}

// 二つの点群間の誤差を最小化する。
// 入力が device_vector になったことと、Isometry3f の代わりに Matrix4f を使っている以外はCPU版と同一。
Eigen::Isometry3f find_transformation_gpu_v1(const thrust::device_vector<Eigen::Vector3f>& target_points, const thrust::device_vector<Eigen::Vector3f>& source_points) {
  // 乱数生成器と姿勢ノイズ分布
  std::mt19937 mt;
  std::normal_distribution<> noise_dist(0.0, 0.1);

  // 現在のベスト姿勢とベスト誤差（最小誤差）
  Eigen::Matrix4f best_T = Eigen::Matrix4f::Identity();
  double best_error = evaluate_error_gpu_v1(target_points, source_points, best_T);

  for (int i = 0; i < 8192; i++) {
    // 適当な姿勢ノイズを生成する。
    Eigen::Matrix4f noise = Eigen::Matrix4f::Identity();
    noise.block<3, 1>(0, 3) << noise_dist(mt), noise_dist(mt), noise_dist(mt);
    noise.block<3, 3>(0, 0) = Eigen::Quaternionf(1.0f, noise_dist(mt), noise_dist(mt), noise_dist(mt)).normalized().toRotationMatrix();

    // 現在の最適姿勢にノイズを加え、誤差を計算する。
    Eigen::Matrix4f T = best_T * noise;
    double error = evaluate_error_gpu_v1(target_points, source_points, T);

    // もし誤差がベスト値より小さければ、ベスト姿勢とベスト誤差を更新する。
    if (error < best_error) {
      best_error = error;
      best_T = T;
    }
  }

  return Eigen::Isometry3f(best_T);
}
