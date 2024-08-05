#include <iostream>
#include <tbb/tbb.h>

int main(int argc, char** argv) {
  tbb::concurrent_vector<int> vec;

  // reserve はスレッドセーフでないので、
  // 必ず最初に単一スレッドしか動いていない状態で呼び出す。
  // その他、resize, shrink_to_fit, swap, clear などもスレッドセーフではない。
  vec.reserve(2);

  // push_back はスレッドセーフ。
  vec.push_back(0);
  vec.push_back(1);

  // 最初の要素のポインタを保持しておく。
  int* ptr0 = &vec[0];

  // grow_by は要素をN個追加し、追加された最初の要素のイテレータを返す。
  auto itr = vec.grow_by(2);
  itr[0] = 2;
  itr[1] = 3;

  // grow_to_at_least は要素数がN以上になるようにベクタを拡張する。
  vec.grow_to_at_least(5);
  vec[4] = 4;

  for (int i = 0; i < vec.size(); i++) {
    std::cout << "vec[" << i << "]" << " = " << vec[i] << std::endl;
  }

  // ベクタが拡張されても各要素のメモリ位置は不変。
  int* ptr0_ = &vec[0];
  std::cout << "&vec[0]=" << ptr0 << " -> " << ptr0_ << std::endl;

  return 0;
}