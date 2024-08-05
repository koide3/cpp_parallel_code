#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

struct transformation_kernel1 {
public:
  __device__ float operator()(const thrust::tuple<const Eigen::Vector3f&, const Eigen::Vector3f&> inputs) const {
    const Eigen::Vector3f& pt1 = thrust::get<0>(inputs);
    const Eigen::Vector3f& pt2 = thrust::get<1>(inputs);
    const Eigen::Vector3f transed = T.linear() * pt1 + T.translation();
    return (pt1 - pt2).squaredNorm();
  }

public:
  Eigen::Isometry3f T;
};

struct transformation_kernel2 {
public:
  __device__ float operator()(int i) const {
    const Eigen::Vector3f& pt1 = inputs1[i];
    const Eigen::Vector3f& pt2 = inputs2[i];
    const Eigen::Vector3f transed = T.linear() * pt1 + T.translation();
    return (pt1 - pt2).squaredNorm();
  }

public:
  const Eigen::Vector3f* inputs1;
  const Eigen::Vector3f* inputs2;
  Eigen::Isometry3f T;
};

int main(int argc, char** argv) {
  const int num_points = 8192;
  thrust::host_vector<Eigen::Vector3f> h_inputs1(num_points);
  thrust::host_vector<Eigen::Vector3f> h_inputs2(num_points);
  for (int i = 0; i < num_points; i++) {
    h_inputs1[i].setRandom();
    h_inputs2[i].setRandom();
  }

  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
  T.linear() = Eigen::Quaternionf::UnitRandom().toRotationMatrix();
  T.translation() = Eigen::Vector3f::Random();

  thrust::device_vector<Eigen::Vector3f> d_inputs1 = h_inputs1;
  thrust::device_vector<Eigen::Vector3f> d_inputs2 = h_inputs2;
  thrust::device_vector<float> d_outputs(num_points);

  transformation_kernel1 kernel1 = {T};
  thrust::transform(
    thrust::cuda::par,
    thrust::make_zip_iterator(d_inputs1.begin(), d_inputs2.begin()),
    thrust::make_zip_iterator(d_inputs1.end(), d_inputs2.end()),
    d_outputs.begin(),
    kernel1);

  transformation_kernel2 kernel2 = {d_inputs1.data().get(), d_inputs2.data().get(), T};
  thrust::transform(thrust::cuda::par, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(num_points), d_outputs.begin(), kernel2);

  thrust::host_vector<float> h_outputs = d_outputs;
}