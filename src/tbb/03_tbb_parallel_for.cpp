#include <mutex>
#include <iostream>
#include <tbb/tbb.h>

int main(int argc, char** argv) {
  // 並列実行される処理を定義。
  const auto task = [&](const tbb::blocked_range<size_t>& range) {
    // blocked_range はスレッドに割り当てられた一定数のイテレーションをひとまとめにしたブロックを表す。
    // ここでは割り当てられた range.begin() から range.end() の範囲のデータを処理する。
    for (size_t i = range.begin(); i != range.end(); i++) {
      std::cout << "i=" << i << " thread=" << std::this_thread::get_id() << std::endl;
    }
  };

  size_t begin = 0;      // 範囲開始値
  size_t end = 16;       // 範囲終了値
  size_t grainsize = 4;  // チャンクサイズ

  // 並列ループ処理を実行する。
  tbb::parallel_for(tbb::blocked_range<size_t>(begin, end, grainsize), task);

  // 上の例と等価な処理の別の書き方。
  // task2 には blocked_range の範囲内の要素を一つずつ取り出したものが与えられる。
  const auto task2 = [](int i) { std::cout << "i=" << i << " thread=" << std::this_thread::get_id() << std::endl; };
  tbb::parallel_for(begin, end, grainsize, task2);

  // コンテナの各要素を並列処理する。
  std::vector<std::string> vec = {"alfa", "bravo", "charlie", "delta", "echo"};
  const auto task3 = [](const std::string& x) { std::cout << "x=" << x << " thread=" << std::this_thread::get_id() << std::endl; };
  tbb::parallel_for_each(vec.begin(), vec.end(), task3);

  // 上の parallel_for_each は下の range-based for と等価の処理を並列実行する。
  // for (const auto& x : vec) {
  //   task3(x);
  // }

  return 0;
}