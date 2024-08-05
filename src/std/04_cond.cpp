#include <deque>
#include <mutex>
#include <thread>
#include <iostream>
#include <condition_variable>

int main(int argc, char** argv) {
  std::mutex task_queue_mutex;            // タスクキューを保護するミューテックス
  std::condition_variable task_queue_cv;  // タスクキューの状態を通知するための条件変数
  std::deque<int> task_queue;             // タスクキュー

  std::atomic_bool kill_switch = false;  // スレッドを終了させるためのフラグ
  std::thread thread([&] {
    // 終了フラグが立つまで無限ループする
    while (!kill_switch) {
      // ミューテックスをロックする
      std::unique_lock<std::mutex> lock(task_queue_mutex);

      // `conditional_variable::wait` は、第2引数の関数が `true` を返すまで待機する
      // この場合は、タスクキューが空でない場合に待機を解除する
      // 待機中はミューテックスがアンロックされる
      task_queue_cv.wait(lock, [&] { return !task_queue.empty(); });

      if (task_queue.empty()) {
        continue;
      }

      // タスクキューからタスクを取り出す
      int task = task_queue.front();
      task_queue.pop_front();
      lock.unlock();

      std::cout << "task " << task << std::endl;
    }
  });

  for (int i = 0; i < 3; i++) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // ミューテックスをロックし、タスクキューにタスクを追加する
    std::lock_guard<std::mutex> lock(task_queue_mutex);
    task_queue.push_back(i);
    std::cout << "push " << i << std::endl;

    // タスクキューにタスクが追加されたことを通知する
    // これをトリガーとしてスレッドが待機解除される
    task_queue_cv.notify_all();
  }

  // 終了フラグを立てて、スレッドを待機解除する
  kill_switch = true;
  task_queue_cv.notify_all();
  thread.join();

  return 0;
}