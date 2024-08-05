#include <atomic>
#include <thread>
#include <iostream>

int main(int argc, char** argv) {
  int counter = 0;
  const auto unsafe_increment = [&counter] {
    for (int i = 0; i < 1000000; ++i) {
      counter++;
    }
  };

  std::thread thread1(unsafe_increment);
  std::thread thread2(unsafe_increment);

  thread1.join();
  thread2.join();

  std::cout << "counter: " << counter << std::endl;

  std::atomic_int atomic_counter = 0;
  const auto safe_increment = [&atomic_counter] {
    for (int i = 0; i < 1000000; ++i) {
      atomic_counter++;
    }
  };

  thread1 = std::thread(safe_increment);
  thread2 = std::thread(safe_increment);

  thread1.join();
  thread2.join();

  std::cout << "atomic_counter: " << atomic_counter << std::endl;

  return 0;
}