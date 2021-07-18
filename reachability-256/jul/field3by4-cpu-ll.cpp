#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include "field3by4lib-cpu.h"


bool get_value(const unsigned char *arr, long long pos) {
    return arr[pos >> 3] & 1 << (pos & 7);
}
void set_value(unsigned char *arr, long long pos, bool value) {
    if (value != get_value(arr, pos))
        arr[pos >> 3] ^= 1 << (pos & 7);
}


const int layers_cnt = 3 * 4 * win_const / 2 + 1;
long long index_dp[13][layers_cnt];

long long encode_inside_layer(const board &b, int sum) {
    // assert(b.sum() == sum);
    long long result = 0;
    int left = 11;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j, --left) {
            if (b.f[i][j] == 0)
                continue;
            result += index_dp[left][sum];
            for (int tile = 2; tile < b.f[i][j]; tile *= 2)
                result += index_dp[left][sum - tile];
            sum -= b.f[i][j];
        }
    }
    // assert(sum == 0);
    return result;
}
long long encode_inside_layer(const board &b) {
    return encode_inside_layer(b, b.sum());
}
board decode_inside_layer(long long id, int sum) {
    board result;
    int left = 11;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j, --left) {
            if (id < index_dp[left][sum]) {
                result.f[i][j] = 0;
                continue;
            }
            id -= index_dp[left][sum];
            int tile = 2;
            while (id >= index_dp[left][sum - tile])
                id -= index_dp[left][sum - tile], tile *= 2;
            result.f[i][j] = tile;
            sum -= tile;
        }
    }
    // assert(sum == 0);
    return result;
}

bool get_value(const unsigned char *const arr[layers_cnt], const board &b) {
    return get_value(arr[b.sum()], encode_inside_layer(b, b.sum()));
}


const int CPU_concurrency = 7;  // works better than 8

void process_layer(unsigned char *const arr[layers_cnt], int sum, long long begin, long long end) {
    for (int position_id = begin; position_id < end; ++position_id) {
        board position(decode_inside_layer(position_id, sum));
        // assert(position.sum() == sum);
        bool winnable = false;
        for (int swipe = 0; swipe < 4; ++swipe) {
            board temp(position);
            bool possible;
            switch (swipe) {
                case 0: possible = temp.swipeLeft(); break;
                case 1: possible = temp.swipeRight(); break;
                case 2: possible = temp.swipeUp(); break;
                case 3: possible = temp.swipeDown(); break;
            }
            if (!possible)
                continue;
            if (temp.won()) {
                winnable = true;
                break;
            }
            bool excellent = true;
            for (int shot = 0; shot < 24; ++shot) {
                board test(temp);
                if (test.addTile(shot)) {
                    int s = test.sum();
                    excellent &= get_value(arr[s], encode_inside_layer(test, s));
                }
            }
            winnable |= excellent;
        }
        set_value(arr[sum], position_id, winnable);
    }
}


int main() {
    system("date");

    report_start("calculating index dp");
    // index_dp[cnt][sum] gives the number of ways to get sum in cnt cells (empty allowed)
    // useful for fast indexation inside layers of constant sum; checked to be at most 150246756
    index_dp[0][0] = 1;
    std::fill(index_dp[0] + 1, index_dp[0] + layers_cnt, 0);
    for (int cnt = 1; cnt <= 12; ++cnt) {
        for (int sum = 0; sum <= 3 * 4 * win_const / 2; ++sum) {
            index_dp[cnt][sum] = index_dp[cnt - 1][sum];
            for (int tile = 2; tile < win_const && tile <= sum; tile *= 2)
                index_dp[cnt][sum] += index_dp[cnt - 1][sum - tile];
        }
    }
    report_finish();

    report_start("allocating dynamic memory (" + itos(std::accumulate(index_dp[12], index_dp[12] + layers_cnt, 0ll) / 8
            + layers_cnt) + ")");
    unsigned char *arr_by_layers[layers_cnt];
    for (int sum = 0; sum < layers_cnt; ++sum)
        arr_by_layers[sum] = new unsigned char[index_dp[12][sum] / 8 + 1];
    report_finish();

    std::thread kernels[CPU_concurrency];
    for (int sum = 3 * 4 * win_const / 2; sum >= 0; --sum) {
        if (index_dp[12][sum] == 0)
            continue;
        report_start("processing layer with sum " + itos(sum));
        long long total_cnt = index_dp[12][sum];
        for (int i = 0; i < CPU_concurrency; ++i)
            kernels[i] = std::thread(process_layer, arr_by_layers, sum, total_cnt / CPU_concurrency * i / 8 * 8,
                    i == CPU_concurrency - 1 ? total_cnt : total_cnt / CPU_concurrency * (i + 1) / 8 * 8);
        for (int i = 0; i < CPU_concurrency; ++i)
            kernels[i].join();
        report_finish();
    }

    std::cout << "FINAL RESULT:\n";
    std::cout << "empty field:\n";
    board empty;
    std::cout << "\t" << get_value(arr_by_layers, empty) << "\n";
    std::cout << "field with just one tile:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "\t";
        for (int j = 0; j < 4; ++j) {
            board b;
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 4; ++l)
                    b.f[k][l] = k == i && l == j ? 2 : 0;
            std::cout << get_value(arr_by_layers, b);
            b.f[i][j] = 4;
            std::cout << get_value(arr_by_layers, b) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "field with two tiles:\n";
    for (int i1 = 0; i1 < 3; ++i1) {
        for (int i2 = 0; i2 < 3; ++i2) {
            for (int j1 = 0; j1 < 4; ++j1) {
                std::cout << "\t";
                for (int j2 = 0; j2 < 4; ++j2) {
                    if (i1 == i2 && j1 == j2) {
                        std::cout << "-- ";
                        continue;
                    }
                    board b;
                    for (int k = 0; k < 3; ++k)
                        for (int l = 0; l < 4; ++l)
                            b.f[k][l] = k == i1 && l == j1 ? 2 : k == i2 && l == j2 ? 2 : 0;
                    std::cout << get_value(arr_by_layers, b);
                    b.f[i2][j2] = 4;
                    std::cout << get_value(arr_by_layers, b) << " ";
                }
            }
            std::cout << "\t\t";
            for (int j1 = 0; j1 < 4; ++j1) {
                std::cout << "\t";
                for (int j2 = 0; j2 < 4; ++j2) {
                    if (i1 == i2 && j1 == j2) {
                        std::cout << "-- ";
                        continue;
                    }
                    board b;
                    for (int k = 0; k < 3; ++k)
                        for (int l = 0; l < 4; ++l)
                            b.f[k][l] = k == i1 && l == j1 ? 4 : k == i2 && l == j2 ? 2 : 0;
                    std::cout << get_value(arr_by_layers, b);
                    b.f[i2][j2] = 4;
                    std::cout << get_value(arr_by_layers, b) << " ";
                }
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }


    system("date");

    reenter_pos:

    std::cout << "Enter position:\n";
    board b;
    for (int i = 0; i < 3; ++i) {
        std::cout << '\t';
        for (int j = 0; j < 4; ++j)
            std::cin >> b.f[i][j];
    }
    std::cout << "winnable: " << get_value(arr_by_layers, b) << "\n";
    if (get_value(arr_by_layers, b))
        goto reenter_pos;

    while (true) {
        reenter_swipe:
        std::cout << "your swipe: ";
        std::string swipe;
        std::cin >> swipe;
        bool possible;
        if (swipe == "left")
            possible = b.swipeLeft();
        else if (swipe == "right")
            possible = b.swipeRight();
        else if (swipe == "up")
            possible = b.swipeUp();
        else if (swipe == "down")
            possible = b.swipeDown();
        else if (swipe == "fine")
            goto reenter_pos;
        else
            goto reenter_swipe;
        if (!possible)
            goto reenter_swipe;
        std::cout << "position after swipe: " << b << "\n";
        std::vector<int> ways;
        for (int i = 0; i < 24; ++i) {
            board test(b);
            if (test.addTile(i) && !get_value(arr_by_layers, test))
                ways.push_back(i);
        }
        if (ways.empty()) {
            std::cout << "ALGORITHM FAILURE.\n\n";
            goto reenter_pos;
        }
        int move = ways[rand() % ways.size()];
        assert(b.addTile(move) && !get_value(arr_by_layers, b));
        std::cout << "position after answer: " << b << "\n";
    }
}
