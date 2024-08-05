#include <omp.h>
#include <vector>
#include <numeric>
#include <list>
#include <iostream>

int main(int argc, char** argv) {
  const size_t num_data = 32;
  std::vector<int> inputs(num_data);
  std::iota(inputs.begin(), inputs.end(), 0);  // inputsを0 ~ num_data-1で埋める

  int sum_squares = 0;

  // 1. 並列化しない場合
  for (int i = 0; i < inputs.size(); i++) {
    sum_squares += inputs[i] * inputs[i];
  }
  std::cout << "sum_squares(serial)=" << sum_squares << std::endl;

  sum_squares = 0;

  // 2. 並列化する場合 (OpenMP reduction)
#pragma omp parallel for reduction(+ : sum_squares)
  for (int i = 0; i < inputs.size(); i++) {
    sum_squares += inputs[i] * inputs[i];
  }
  std::cout << "sum_squares(omp reduction)=" << sum_squares << std::endl;

  // 3. 並列化する場合 (self reduction)
  std::vector<int> sum_squares_per_thread(omp_get_max_threads(), 0);  // 各スレッドごとの積算結果
#pragma omp parallel for reduction(+ : sum_squares)
  for (int i = 0; i < inputs.size(); i++) {
    sum_squares_per_thread[omp_get_thread_num()] += inputs[i] * inputs[i];
  }

  // 各スレッドごとの結果を合計する。
  for (int i = 1; i < sum_squares_per_thread.size(); i++) {
    sum_squares_per_thread[0] += sum_squares_per_thread[i];
  }
  std::cout << "sum_squares(self reduction)=" << sum_squares_per_thread[0] << std::endl;

  return 0;
}