#include <iostream>
#include <tbb/tbb.h>

int main(int argc, char** argv) {
  // A. データフローグラフ構築

  // データ処理に使用するデータ構造
  struct MyData {
    int i;            // データID
    std::string str;  // IDを文字列に変換したもの
  };

  // グラフインスタンス
  tbb::flow::graph graph;

  // 入力ノード
  tbb::flow::broadcast_node<int> input_node(graph);

  // 処理ノード
  // ここでは整数から文字列への変換を行う。
  // あとで順番をソートしなおすために出力は元の数値と変換後の文字列のペアとする。
  tbb::flow::function_node<int, MyData> convert_node(
    graph,                 // グラフインスタンス
    tbb::flow::unlimited,  // 最大並列数(unlimited = 無制限、serial = 1 = 並列実行しない)
    // このノードが行う処理内容。
    [](int i) -> MyData {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << "processing " << i << " thread=" << std::this_thread::get_id() << std::endl;
      return MyData{i, "data=" + std::to_string(i)};
    });

  // データのIDによってデータを再整列する。
  tbb::flow::sequencer_node<MyData> sequencer_node(graph, [](const MyData& data) { return data.i; });

  // 出力ノード
  tbb::flow::function_node<MyData> print_node(
    graph,              //
    tbb::flow::serial,  // 並列処理を無効にして、一つずつ出力する。
    [](const MyData& data) { std::cout << data.str << std::endl; });

  // ノード間にエッジ（データ接続）を作成する。
  tbb::flow::make_edge(input_node, convert_node);
  tbb::flow::make_edge(convert_node, sequencer_node);
  tbb::flow::make_edge(sequencer_node, print_node);

  // B. データフローグラフの実行

  // 入力ノードにデータを流し込む。
  for (int i = 0; i < 10; i++) {
    input_node.try_put(i);
  }

  // グラフ上の全データ処理が終了するまで待つ。
  graph.wait_for_all();

  return 0;
}