#include <vector>
#include <atomic>
#include <thread>
#include <iostream>
#include <tbb/tbb.h>

// 基本的な使用方法
void basic_example() {
  tbb::concurrent_queue<int> queue;

  // push でデータを追加する。
  queue.push(1);

  int val;

  // try_pop はデータを取り出すことに成功した場合、
  // val に取り出されたデータを格納し true を返す。
  if (queue.try_pop(val)) {
    std::cout << "popped val=" << val << std::endl;
  }
  // キューが空の場合、データの取り出しに失敗し、falseを返す。
  else {
    std::cout << "no data in queue" << std::endl;
  }

  // もう一回、データを取り出そうとすると、キューがからなので失敗する。
  if (queue.try_pop(val)) {
    std::cout << "popped val=" << val << std::endl;
  } else {
    std::cout << "no data in queue" << std::endl;
  }
}

// 並列データ生成＆処理
void producer_consumer() {
  std::atomic_bool finished = false;       // データ終了フラグ
  tbb::concurrent_queue<int> input_queue;  // 入力データキュー

  // 処理スレッドの生成
  const int num_threads = 4;
  std::vector<std::thread> threads(num_threads);
  std::ranges::generate(threads, [&] {
    return std::thread(
      // 各スレッドの処理内容。
      [&] {
        // 終了フラグが立っていて、入力キューが空なら処理を終了する。
        while (!(finished && input_queue.empty())) {
          // データをキューから取り出す。
          int val;
          if (!input_queue.try_pop(val)) {
            // 取り出しに失敗した（＝キューが空）場合、
            // タイムスライスを破棄（CPUを食いつぶすのを防ぐ）して最初に戻る。
            std::this_thread::yield();
            continue;
          }

          // 取り出したデータを処理する。
          std::cout << "val=" << val << " thread=" << std::this_thread::get_id() << std::endl;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      });
  });

  // キューにデータを追加していく。
  for (int i = 0; i < 10; i++) {
    input_queue.push(i);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // 終了フラグを立て、全てのスレッドの処理が終了する（キュー内の全データが処理される）のを待つ。
  finished = true;
  for (auto& thread : threads) {
    thread.join();
  }
}

int main(int argc, char** argv) {
  basic_example();
  producer_consumer();
}