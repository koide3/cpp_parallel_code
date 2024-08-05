#include <random>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thrust/device_vector.h>

#include <stopwatch.hpp>
#include <read_points.hpp>
#include <cuda_check_error.hpp>

Eigen::Isometry3f find_transformation(const std::vector<Eigen::Vector3f>& target_points, const std::vector<Eigen::Vector3f>& source_points);
Eigen::Isometry3f find_transformation_gpu_v1(const thrust::device_vector<Eigen::Vector3f>& target_points, const thrust::device_vector<Eigen::Vector3f>& source_points);
Eigen::Isometry3f find_transformation_gpu_v2(const thrust::device_vector<Eigen::Vector3f>& target_points, const thrust::device_vector<Eigen::Vector3f>& source_points);
Eigen::Isometry3f find_transformation_gpu_v3(const thrust::device_vector<Eigen::Vector3f>& target_points, const thrust::device_vector<Eigen::Vector3f>& source_points);
Eigen::Isometry3f find_transformation_gpu_v4(const thrust::device_vector<Eigen::Vector3f>& target_points, const thrust::device_vector<Eigen::Vector3f>& source_points);

int main(int argc, char** argv) {
  auto points_4f = read_ply("data/target.ply");
  std::vector<Eigen::Vector3f> target_points;
  for (int i = 0; i < 5; i++) {
    std::transform(points_4f.begin(), points_4f.end(), std::back_inserter(target_points), [](const Eigen::Vector4f& p) { return p.head<3>(); });
  }

  std::cout << "target_points=" << target_points.size() << std::endl;

  Eigen::Isometry3f T_target_source = Eigen::Isometry3f::Identity();
  T_target_source.linear() = Eigen::AngleAxisf(0.5f, Eigen::Vector3f::Random().normalized()).toRotationMatrix();
  T_target_source.translation() = Eigen::Vector3f::Random();

  Eigen::Isometry3f T_source_target = T_target_source.inverse();

  std::vector<Eigen::Vector3f> source_points(target_points.size());
  std::transform(target_points.begin(), target_points.end(), source_points.begin(), [&](const Eigen::Vector3f& p) { return T_source_target * p; });

  Stopwatch sw;
  sw.start();
  Eigen::Isometry3f estimated_T_target_source = find_transformation(target_points, source_points);
  sw.stop();

  std::cout << "cpu=" << sw.msec() << "msec" << std::endl;
  std::cout << "--- T_target_source ---" << std::endl << T_target_source.matrix() << std::endl;
  std::cout << "--- estimated_T_target_source ---" << std::endl << estimated_T_target_source.matrix() << std::endl;

  thrust::device_vector<Eigen::Vector3f> d_target_points = target_points;
  thrust::device_vector<Eigen::Vector3f> d_source_points = source_points;

  sw.start();
  Eigen::Isometry3f estimated_T_target_source_v1 = find_transformation_gpu_v1(d_target_points, d_source_points);
  sw.stop();

  std::cout << "gpu_v1=" << sw.msec() << "msec" << std::endl;
  std::cout << "--- estimated_T_target_source_v1 ---" << std::endl << estimated_T_target_source_v1.matrix() << std::endl;

  sw.start();
  Eigen::Isometry3f estimated_T_target_source_v2 = find_transformation_gpu_v2(d_target_points, d_source_points);
  sw.stop();

  std::cout << "gpu_v2=" << sw.msec() << "msec" << std::endl;
  std::cout << "--- estimated_T_target_source_v2 ---" << std::endl << estimated_T_target_source_v2.matrix() << std::endl;

  sw.start();
  Eigen::Isometry3f estimated_T_target_source_v3 = find_transformation_gpu_v3(d_target_points, d_source_points);
  sw.stop();

  std::cout << "gpu_v3=" << sw.msec() << "msec" << std::endl;
  std::cout << "--- estimated_T_target_source_v3 ---" << std::endl << estimated_T_target_source_v3.matrix() << std::endl;

  sw.start();
  Eigen::Isometry3f estimated_T_target_source_v4 = find_transformation_gpu_v4(d_target_points, d_source_points);
  sw.stop();

  std::cout << "gpu_v4=" << sw.msec() << "msec" << std::endl;
  std::cout << "--- estimated_T_target_source_v4 ---" << std::endl << estimated_T_target_source_v4.matrix() << std::endl;

  return 0;
}