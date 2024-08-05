#include <vector>
#include <numeric>
#include <list>
#include <iostream>
#include <stopwatch.hpp>

int main(int argc, char** argv) {
  const int num_data = 32;
  std::vector<int> inputs(num_data);
  std::iota(inputs.begin(), inputs.end(), 0);
  std::vector<int> outputs(num_data);

  Stopwatch sw;

  // 1. 並列化しない場合
  sw.start();
  for (int i = 0; i < outputs.size(); i++) {
    outputs[i] = inputs[i] * inputs[i];
  }
  sw.stop();
  std::cout << "time=" << sw.msec() << "[msec]" << std::endl;

  // 2. 並列化する場合 (OpenMPは初回実行時に初期化で時間がかかる。)
  sw.start();
#pragma omp parallel for
  for (int i = 0; i < outputs.size(); i++) {
    outputs[i] = inputs[i] * inputs[i];
  }
  sw.stop();
  std::cout << "time=" << sw.msec() << "[msec]" << std::endl;

  // 3. 並列化する場合 (2回目以降は高速に処理がかかる。)
  sw.start();
#pragma omp parallel for
  for (int i = 0; i < outputs.size(); i++) {
    outputs[i] = inputs[i] * inputs[i];
  }
  sw.stop();
  std::cout << "time=" << sw.msec() << "[msec]" << std::endl;

  // 結果の表示
  for (int i = 0; i < outputs.size(); i++) {
    std::cout << inputs[i] << " * " << inputs[i] << " = " << outputs[i] << std::endl;
  }

  return 0;
}