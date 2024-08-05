#include <string>
#include <vector>
#include <iostream>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

void normal_iterator() {
  std::vector<int> data = {0, 1, 2, 3, 4};

  for (auto itr = data.begin(); itr != data.end(); itr++) {
    std::cout << *itr << " ";
  }
  std::cout << std::endl;
}

void transform_iterator() {
  std::vector<int> data = {0, 1, 2, 3, 4};

  auto func = [](int x) { return 2 * x; };
  auto begin = thrust::make_transform_iterator(data.begin(), func);

  for (auto itr = begin; itr != begin + data.size(); itr++) {
    std::cout << *itr << " ";
  }
  std::cout << std::endl;
}

void transform_iterator2() {
  std::vector<int> data = {0, 1, 2, 3, 4};

  auto func = [](int x) { return "data=" + std::to_string(2 * x); };
  auto begin = thrust::make_transform_iterator(data.begin(), func);

  for (auto itr = begin; itr != begin + data.size(); itr++) {
    std::cout << *itr << " ";
  }
  std::cout << std::endl;
}

void counting_iterator() {
  auto begin = thrust::make_counting_iterator<int>(0);
  for (auto itr = begin; itr != begin + 8; itr++) {
    std::cout << *itr << " ";
  }
  std::cout << std::endl;
}

void zip_iterator() {
  std::vector<int> data1 = {0, 1, 2, 3, 4};
  std::vector<char> data2 = {'a', 'b', 'c', 'd', 'e'};

  auto begin = thrust::make_zip_iterator(data1.begin(), data2.begin());
  auto end = begin + data1.size();
  for (auto itr = begin; itr != end; itr++) {
    std::cout << thrust::get<0>(*itr) << ":" << thrust::get<1>(*itr) << " ";
  }
  std::cout << std::endl;
}

void combine_iterators() {
  std::vector<std::string> data = {"alpha", "bravo", "charlie", "delta", "echo"};

  auto func = [&](int i) { return data[i]; };

  auto begin = thrust::make_zip_iterator(  //
    thrust::make_counting_iterator<int>(0),
    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), func));

  for (auto itr = begin; itr != begin + data.size(); itr++) {
    int i = thrust::get<0>(*itr);
    std::string str = thrust::get<1>(*itr);

    std::cout << "i=" << i << ":" << str << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  normal_iterator();
  transform_iterator();
  transform_iterator2();
  counting_iterator();
  zip_iterator();
  combine_iterators();

  return 0;
}