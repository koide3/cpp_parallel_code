#pragma once

#include <chrono>

struct Stopwatch {
public:
  Stopwatch() { start(); }

  void start() { t1 = std::chrono::high_resolution_clock::now(); }
  void stop() { t2 = std::chrono::high_resolution_clock::now(); }
  void lap() {
    t1 = t2;
    t2 = std::chrono::high_resolution_clock::now();
  }

  double msec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6; }
  double sec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9; }

public:
  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
};