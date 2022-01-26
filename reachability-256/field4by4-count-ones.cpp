#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include "field4by4lib-cpu.h"


void check_filename_sanity(const std::string& filename) {
  if (filename.substr(0, 6) != "ulayer") {
    throw std::invalid_argument("filename must start with \"ulayer\"");
  }
  if (filename.length() != 14) {
    throw std::invalid_argument("filename length must be 14 (\"ulayer????.dat\")");
  }
  if (filename.substr(10, 4) != ".dat") {
    throw std::invalid_argument("filename must end with \".dat\"");
  }
}

bool file_processed(std::string filename) {
  std::ifstream database("count-ones-db.txt");
  if (!database.is_open()) {
    throw std::runtime_error("no count-ones-db.txt");
  }
  check_filename_sanity(filename);
  filename = filename.substr(6, 4);
  int sum;
  std::stringstream ss1, ss2;
  ss1 << filename;
  ss1 >> sum;
  ss2 << sum;
  std::string sumstr, line;
  ss2 >> sumstr;
  while (getline(database, line)) {
    if (line.substr(0, line.find('\t')) == sumstr)
      return true;
  }
  return false;
}

struct entry {
  int layer_sum;
  long long cnt_ones;
};

entry count_ones(std::string filename) {
  check_filename_sanity(filename);
  uint8_t ones_cnt[256];
  for (uint32_t u = 0; u < 256; ++u) {
    ones_cnt[u] = __builtin_popcount(u);
  }
  int layer_sum;
  filename = filename.substr(6, 4);
  std::stringstream ss;
  ss << filename;
  ss >> layer_sum;
  unsigned char *layer;
  layer = new unsigned char[index_dp[16][layer_sum] / 8 + 1];
  read_arr_from_disk(index_dp[16][layer_sum] / 8 + 1, layer, "arrays4by4-256/ulayer" + itos(layer_sum, 4) + ".dat");
  long long cnt_ones = 0;
  for (long long i = 0; i < index_dp[16][layer_sum] / 8 + 1; ++i) {
    cnt_ones += ones_cnt[layer[i]];
  }
  delete[] layer;
  return {layer_sum, cnt_ones};
}

void add_entry(const entry e) {
  std::ofstream database("count-ones-db.txt", std::ios::app);
  database << e.layer_sum << "\t" << e.cnt_ones << "\n";
}

int main() {
  fill_index_dp();
  std::string path = "arrays4by4-256";
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    std::string filename = entry.path();
    filename = filename.substr(15);  // strip "arrays4by4-256/"
    if (filename.ends_with(".gz") || file_processed(filename))
      continue;
    std::cout << "processing: " << filename << "...";
    std::cout.flush();
    add_entry(count_ones(filename));
    std::cout << "\t\tdone" << std::endl;
  }
}
