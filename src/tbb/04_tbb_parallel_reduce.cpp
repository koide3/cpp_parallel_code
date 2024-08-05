#include <iostream>
#include <tbb/tbb.h>

struct Sum {
  // コンストラクタ
  // my_data は積算される数値配列
  // my_sum は現在までに積算された数値（最初は0）
  Sum(double* my_data) : my_data(my_data), my_sum(0.0) {}

  // 積算処理を別スレッドに分割するときに呼ばれるコンストラクタ
  Sum(Sum& x, tbb::split) : my_data(x.my_data), my_sum(0.0) {}

  // 加算処理
  void operator()(const tbb::blocked_range<size_t>& r) {
    // 積算値をローカル変数として保持する。
    // 直接 my_sum に加算するよりローカル変数として処理した方がキャッシュ効率が良くなる。
    double sum = my_sum;

    // 割り当てられた範囲のデータを積算する。
    for (size_t i = r.begin(); i != r.end(); i++) {
      sum += my_data[i] * my_data[i];
    }

    // 積算した結果を my_sum に適用する。
    my_sum = sum;
  }

  // 分割して積算された結果を統合する。
  void join(const Sum& y) { my_sum += y.my_sum; }

  const double* my_data;  // 積算される数値列
  double my_sum;          // 積算された数値
};

int main(int argc, char** argv) {
  std::vector<double> values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

  // 並列リダクションを実行する。
  Sum sum(values.data());
  tbb::parallel_reduce(tbb::blocked_range<size_t>(0, values.size()), sum);

  std::cout << "sum=" << sum.my_sum << std::endl;

  return 0;
}