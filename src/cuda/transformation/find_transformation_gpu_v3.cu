#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <cuda_check_error.hpp>

// 点間の二乗距離を計算するカーネル。
// ★ v2 と機能は同一だが、zip_iterator ではなく普通のインデックス番号としてデータを受け取る。
struct evaluate_error_kernel_v3 {
  __device__ float operator()(int i) const {
    const Eigen::Vector3f target = target_points[i];  // ★ i 番目のターゲット点
    const Eigen::Vector3f source = source_points[i];  // ★ i 番目のソース点
    return (target - (T.block<3, 3>(0, 0) * source + T.block<3, 1>(0, 3))).squaredNorm();
  }

  const Eigen::Vector3f* target_points;  // ★ ターゲット点配列
  const Eigen::Vector3f* source_points;  // ★ ソース点配列
  Eigen::Matrix4f T;
};

// 二つの点群間の二乗距離を計算する。
// ★ v2から引数に以下が増えている。
// d_result             : GPU上で reduce 結果を格納する領域
// h_result             : CPU上で reduce 結果を受け取る領域（pinned メモリ）
// d_temp_storage       : reduce 計算用のGPU上の計算バッファ
// d_temp_storage_bytes : 計算バッファのサイズ
float evaluate_error_gpu_v3(
  const thrust::device_vector<Eigen::Vector3f>& target_points,
  const thrust::device_vector<Eigen::Vector3f>& source_points,
  const Eigen::Matrix4f& T,
  cudaStream_t stream,
  float* d_result,
  float* h_result,
  void** d_temp_storage,
  size_t& d_temp_storage_bytes) {
  // ★ 距離計算カーネルを作成。ターゲット点・ソース点をGPU上の生ポインタとして渡す。
  evaluate_error_kernel_v3 kernel{target_points.data().get(), source_points.data().get(), T};

  // ★ インデックス番号を取り、距離計算結果を返す transform イテレータを作成する。
  cub::TransformInputIterator<
    float,                          // 出力は float
    evaluate_error_kernel_v3,       // 変換処理
    thrust::counting_iterator<int>  // 入力は counting_iterator<int>
    >
    d_begin(thrust::make_counting_iterator<int>(0), kernel);

  // ★ DeviceReduce::Sum に必要な計算バッファサイズを計算する。
  size_t required_temp_storage_bytes = 0;
  check_error << cub::DeviceReduce::Sum(
    static_cast<void*>(nullptr),  // 計算バッファ (nullptr でバッファサイズ計算モード)
    required_temp_storage_bytes,  // [out] 計算バッファサイズ
    d_begin,                      // 先頭入力データ(GPUメモリ)
    d_result,                     // 計算結果の格納先(GPUメモリ)
    target_points.size(),         // 入力データ数
    stream                        // 処理を行うストリーム
  );

  // ★ 必要な計算バッファサイズが、今のバッファサイズより大きければ新しくバッファを確保しなおす。
  // ★ なお、cub::DeviceReduce が必要とするバッファサイズは結果データのサイズと入力データ数で決まるため、
  // ★ 今回の場合は必要なバッファサイズは毎回同じになる。
  if (required_temp_storage_bytes > d_temp_storage_bytes) {
    // ★ 必要なサイズになるようにバッファを確保しなおす。
    check_error << cudaFreeAsync(*d_temp_storage, stream);
    check_error << cudaMallocAsync(d_temp_storage, required_temp_storage_bytes, stream);
    d_temp_storage_bytes = required_temp_storage_bytes;
  }

  // ★ DeviceReduce::Sum を実行する。
  // ★ 第一引数が使用する計算バッファとバッファサイズになっている以外は最初の呼び出しと同一。
  check_error << cub::DeviceReduce::Sum(*d_temp_storage, d_temp_storage_bytes, d_begin, d_result, target_points.size(), stream);

  // ★ 計算結果をCPU上の h_result に転送する。
  // ★ h_result は pinned メモリで非同期転送が行われるため、明示的にストリームに対する同期を行う。
  // ★ 関数を通してこの一点のみで同期が行われる。
  check_error << cudaMemcpyAsync(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost, stream);
  check_error << cudaStreamSynchronize(stream);

  return *h_result;
}

// 二つの点群間の誤差を最小化する。
Eigen::Isometry3f find_transformation_gpu_v3(const thrust::device_vector<Eigen::Vector3f>& target_points, const thrust::device_vector<Eigen::Vector3f>& source_points) {
  // ストリームを作成する。
  cudaStream_t stream;
  check_error << cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // ★ reduce の計算結果を格納・転送するためのGPU・CPUメモリを確保する。
  // ★ CPU上のメモリは非同期転送のために `cudaMallocHost` を使って特殊な pinned メモリとして確保する。
  float* d_result;
  float* h_result;
  check_error << cudaMallocAsync(&d_result, sizeof(float), stream);
  check_error << cudaMallocHost(&h_result, sizeof(float));

  // ★ reduce の計算バッファ(GPU上)。
  // ★ この段階では必要なサイズがわからないのでサイズ0としておいて、evaluate_error_gpu_v3 内で確保する。
  void* d_temp_storage = nullptr;
  size_t d_temp_storage_bytes = 0;

  // 乱数生成器と姿勢ノイズ分布
  std::mt19937 mt;
  std::normal_distribution<> noise_dist(0.0, 0.1);

  // 現在のベスト姿勢とベスト誤差（最小誤差）
  Eigen::Matrix4f best_T = Eigen::Matrix4f::Identity();
  double best_error = evaluate_error_gpu_v3(target_points, source_points, best_T, stream, d_result, h_result, &d_temp_storage, d_temp_storage_bytes);

  for (int i = 0; i < 8192; i++) {
    // 適当な姿勢ノイズを生成する。
    Eigen::Matrix4f noise = Eigen::Matrix4f::Identity();
    noise.block<3, 1>(0, 3) << noise_dist(mt), noise_dist(mt), noise_dist(mt);
    noise.block<3, 3>(0, 0) = Eigen::Quaternionf(1.0f, noise_dist(mt), noise_dist(mt), noise_dist(mt)).normalized().toRotationMatrix();

    // 現在の最適姿勢にノイズを加え、誤差を計算する。
    Eigen::Matrix4f T = best_T * noise;

    double error = evaluate_error_gpu_v3(target_points, source_points, T, stream, d_result, h_result, &d_temp_storage, d_temp_storage_bytes);

    // もし誤差がベスト値より小さければ、ベスト姿勢とベスト誤差を更新する。
    if (error < best_error) {
      best_error = error;
      best_T = T;
    }
  }

  // ★ ストリームとメモリを破棄する。
  check_error << cudaFreeHost(h_result);
  check_error << cudaFreeAsync(d_result, stream);
  check_error << cudaFreeAsync(d_temp_storage, stream);
  check_error << cudaStreamDestroy(stream);

  return Eigen::Isometry3f(best_T);
}
