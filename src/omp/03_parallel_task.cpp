#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

// 再帰的マージソート処理の実装
void merge_sort_omp_impl(size_t* first, size_t* last) {
  const size_t n = std::distance(first, last);  // アイテム数
  if (n < 256) {
    // アイテム数が少ない場合は分割をやめてシリアルにソートする。
    std::sort(first, last);
    return;
  }

  auto center = first + n / 2;

  // 前半と後半を並列にソートする。
#pragma omp task
  merge_sort_omp_impl(first, center);

#pragma omp task
  merge_sort_omp_impl(center, last);

  // 前半と後半の処理を待ち、マージを行う。
#pragma omp taskwait
  std::inplace_merge(first, center, last);
}

// parallel task による並列マージソート
void merge_sort_omp(size_t* first, size_t* last) {
#pragma omp parallel  // 並列処理セクションを定義する。
  {
#pragma omp single nowait  // 一つのスレッドから開始する。
    {
      // 再帰的にマージソートを呼び出す。
      merge_sort_omp_impl(first, last);
    }
  }
}

int main(int argc, char** argv) {
  std::vector<size_t> data(8192 * 32);

  std::mt19937 mt;
  std::ranges::generate(data, [&] { return mt(); });

  merge_sort_omp(data.data(), data.data() + data.size());

  std::cout << "is_sorted=" << std::ranges::is_sorted(data) << std::endl;

  return 0;
}