#include <thread>
#include <iostream>

int main(int argc, char** argv) {
  // バックグラウンドで実行する処理。ラムダ式でなく普通の関数を渡してもよい。
  const auto func = [] {
    std::cout << "start " << std::this_thread::get_id() << std::endl;  // 開始時にスレッドIDを表示
    std::this_thread::sleep_for(std::chrono::seconds(1));              // 1秒スリープ
    std::cout << "done " << std::this_thread::get_id() << std::endl;   // 終了時にスレッドIDを表示
  };

  std::cout << "start threads" << std::endl;
  // スレッドを２つ起ち上げてそれぞれで func() を実行する
  std::thread thread1(func);
  std::thread thread2(func);

  // thread1, thread2 の処理が終わるのを待つ
  std::cout << "wait for the threads to be terminated" << std::endl;
  thread1.join();
  thread2.join();
  std::cout << "joined" << std::endl;

  return 0;
}