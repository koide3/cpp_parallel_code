#include <iostream>
#include <tbb/tbb.h>

int main(int argc, char** argv) {
  // 3つの処理(A, B, C)を並列に実行する。
  tbb::parallel_invoke(
    // 処理A
    [] {
      std::cout << "taskA thread=" << std::this_thread::get_id() << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    },
    // 処理B
    [] {
      std::cout << "taskB thread=" << std::this_thread::get_id() << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    },
    // 処理C
    [] {
      std::cout << "taskC thread=" << std::this_thread::get_id() << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    });

  return 0;
}