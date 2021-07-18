#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

long long factorial(int n) {
    static std::vector<long long> v(1, 1);
    if (n < v.size())
        return v[n];
    long long f = factorial(n - 1), ans = f * n;
    v.push_back(ans);
    return ans;
}

std::vector<int> temp;
long long max_arrangement_size;
std::vector<std::vector<int>> arrangements;

long long cnt_positions_with_sum(int field_size, int tiles, int rest_cnt, int rest_sum, std::vector<int>& cnt_by_tile) {
                // tiles = 7 means 2, 4, ..., 128 available
    /*if (rand() % 1 == 0) {
        std::cout << "field_size = " << field_size << ", tiles = " << tiles << ", rest_cnt = " << rest_cnt << ", rest_sum = " << rest_sum << ", ";
        std::cout << "cnt_by_tile: ";
        for (int x : cnt_by_tile)
            std::cout << x << ' ';
        std::cout << std::endl;
    }*/
    if (cnt_by_tile.size() == tiles) {
        if (rest_sum > 0)
            return 0;
        long long cnt = factorial(field_size) / factorial(rest_cnt);
        for (int x: cnt_by_tile)
            cnt /= factorial(x);
        arrangements.push_back(cnt_by_tile);
        max_arrangement_size = std::max(max_arrangement_size, cnt);
        return cnt;
    }
    long long result = 0;
    for (int k = 0; k <= rest_cnt && k << cnt_by_tile.size() + 1 <= rest_sum; ++k) {
        std::cout.flush();
        cnt_by_tile.push_back(k);
        result += cnt_positions_with_sum(field_size, tiles, rest_cnt - k, rest_sum - (k << cnt_by_tile.size()), cnt_by_tile);
        cnt_by_tile.pop_back();
    }
    return result;
}

int main() {
    int field_size, tiles;
    std::cout << "field size: ";
    std::cin >> field_size;
    std::cout << "tiles up to 2**";
    std::cin >> tiles;
    long long max_layer_size = -1, total_size = 0;
    for (int sum = 0; sum <= field_size << tiles; sum += 2) {
        std::cout << "layer " << sum << ", size ";
        long long layer_size = cnt_positions_with_sum(field_size, tiles, field_size, sum, temp);
        max_layer_size = std::max(max_layer_size, layer_size);
        total_size += layer_size;
        std::cout << layer_size << '\n';
    }
    std::cout << "total size: " << total_size << '\n';
    std::cout << "max layer size: " << max_layer_size << '\n';
    std::cout << "arrangements: " << arrangements.size() << '\n';
    std::cout << "max arrangement size: " << max_arrangement_size << '\n';
}