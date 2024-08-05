#include <random>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

// 二つの点群間の誤差総和を求める。 sum(|target_points - T * source_points|^2)
double evaluate_error(const std::vector<Eigen::Vector3f>& target_points, const std::vector<Eigen::Vector3f>& source_points, const Eigen::Isometry3f& T) {
  double sum_errors = 0.0;
  for (int i = 0; i < target_points.size(); i++) {
    // ソース点を移動して、ターゲット点との二乗誤差を求める。
    sum_errors += (target_points[i] - T * source_points[i]).squaredNorm();
  }
  return sum_errors;
}

// 二つの点群間の誤差を最小化する。
Eigen::Isometry3f find_transformation(const std::vector<Eigen::Vector3f>& target_points, const std::vector<Eigen::Vector3f>& source_points) {
  // 乱数生成器と姿勢ノイズ分布
  std::mt19937 mt;
  std::normal_distribution<> noise_dist(0.0, 0.1);

  // 現在のベスト姿勢とベスト誤差（最小誤差）
  Eigen::Isometry3f best_T = Eigen::Isometry3f::Identity();
  double best_error = evaluate_error(target_points, source_points, best_T);

  for (int i = 0; i < 8192; i++) {
    // 適当な姿勢ノイズを生成する。
    Eigen::Isometry3f noise = Eigen::Isometry3f::Identity();
    noise.translation() << noise_dist(mt), noise_dist(mt), noise_dist(mt);
    noise.linear() = Eigen::Quaternionf(1.0f, noise_dist(mt), noise_dist(mt), noise_dist(mt)).normalized().toRotationMatrix();

    // 現在の最適姿勢にノイズを加え、誤差を計算する。
    Eigen::Isometry3f T = best_T * noise;
    double error = evaluate_error(target_points, source_points, T);

    // もし誤差がベスト値より小さければ、ベスト姿勢とベスト誤差を更新する。
    if (error < best_error) {
      best_error = error;
      best_T = T;
    }
  }

  return best_T;
}
