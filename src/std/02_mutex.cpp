#include <mutex>
#include <thread>
#include <iostream>
#include <stopwatch.hpp>

void unsafe() {
  int counter = 0;

  const auto unsafe_increment = [&] {
    for (int i = 0; i < 1000000; ++i) {
      // 複数のスレッドが同時にインクリメントするとデータ競合が発生する。
      counter++;
    }
  };

  std::thread thread1(unsafe_increment);
  std::thread thread2(unsafe_increment);

  thread1.join();
  thread2.join();

  std::cout << "counter: " << counter << std::endl;
}

void safe_with_mutex() {
  int counter = 0;
  std::mutex mutex;  // counter の排他処理用ミューテックス

  const auto safe_increment = [&] {
    for (int i = 0; i < 1000000; ++i) {
      // ミューテックスをロックする。
      std::lock_guard<std::mutex> lock(mutex);
      // このインクリメント処理は複数のスレッドが同時に実行されることがない。
      counter++;
    }
  };

  std::thread thread1(safe_increment);
  std::thread thread2(safe_increment);

  thread1.join();
  thread2.join();

  std::cout << "counter: " << counter << std::endl;
}

void safe_with_atomic() {
  std::atomic_int counter = 0;

  const auto safe_increment = [&] {
    for (int i = 0; i < 1000000; ++i) {
      // アトミック操作はデータ競合が発生しない。
      int old = counter += 1;
    }
  };

  std::thread thread1(safe_increment);
  std::thread thread2(safe_increment);

  thread1.join();
  thread2.join();

  std::cout << "counter: " << counter << std::endl;
}

int main(int argc, char** argv) {
  Stopwatch sw;

  sw.start();
  unsafe();
  sw.stop();
  std::cout << "unsafe=" << sw.msec() << "[msec]" << std::endl;

  sw.start();
  safe_with_mutex();
  sw.stop();
  std::cout << "safe_with_mutex=" << sw.msec() << "[msec]" << std::endl;

  sw.start();
  safe_with_atomic();
  sw.stop();
  std::cout << "safe_with_atomic=" << sw.msec() << "[msec]" << std::endl;

  return 0;
}