#include <future>
#include <thread>
#include <iostream>
#include <algorithm>

// std::async
void async() {
  // `std::async` 引数として与えた関数を非同期（あるいは遅延）に評価する
  std::future<int> result1 = std::async(std::launch::async, [] {
    // この処理は別スレッドで非同期に実行される
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 1;
  });

  std::future<int> result2 = std::async(std::launch::async, [] {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 2;
  });

  // `get()` を呼ぶと非同期処理が終了するまで待って結果を取得する
  // 待機したくない場合は result1.wait_for(std::chrono::milliseconds(0)) などを使う
  std::cout << "result1: " << result1.get() << std::endl;
  std::cout << "result2: " << result2.get() << std::endl;
}

int main(int argc, char** argv) {
  async();
  return 0;
}