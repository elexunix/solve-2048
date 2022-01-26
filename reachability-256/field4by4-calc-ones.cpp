#include <iostream>
#include "field4by4lib-cpu.h"


int main() {
  fill_index_dp();
  uint8_t ones_cnt[256];
  for (uint32_t u = 0; u < 256; ++u) {
    ones_cnt[u] = __builtin_popcount(u);
  }
  int layer_sum;
  unsigned char *layer;
  while (std::cin >> layer_sum) {
    layer = new unsigned char[index_dp[16][layer_sum] / 8 + 1];
    read_arr_from_disk(index_dp[16][layer_sum] / 8 + 1, layer, "arrays4by4-256/ulayer" + itos(layer_sum, 4) + ".dat");
    long long cnt_ones = 0;
    for (long long i = 0; i < index_dp[16][layer_sum] / 8 + 1; ++i) {
      cnt_ones += ones_cnt[layer[i]];
    }
    std::cout << cnt_ones << "\n";
    delete[] layer;
  }
}
