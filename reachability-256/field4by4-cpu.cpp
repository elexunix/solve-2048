#include <algorithm>
#include <bits/stdc++.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <ostream>
#include <thread>
#include <vector>
#include "field4by4lib-cpu.h"


void process_user_layer(unsigned char *const user_layers[layers_cnt], const unsigned char *const hater_layers[layers_cnt],
                int sum, long long begin, long long end) {
    for (long long position_id = begin; position_id < end; ++position_id) {
        board position(decode_inside_layer(position_id, sum));
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
            assert(temp.sum() == sum);
            winnable |= possible && (temp.won() || get_value(hater_layers, temp, sum));
        }
        set_value(user_layers[sum], position_id, winnable);
    }
}
void process_hater_layer(const unsigned char *const user_layers[layers_cnt], unsigned char *const hater_layers[layers_cnt],
                int sum, long long begin, long long end) {
    for (long long position_id = begin; position_id < end; ++position_id) {
        board position(decode_inside_layer(position_id, sum));
        bool winnable = true;
        for (int shot = 0; shot < 32; ++shot) {
            board temp(position);
            winnable &= !temp.addTile(shot) || get_value(user_layers, temp);
        }
        set_value(hater_layers[sum], position_id, winnable);
    }
}


int main() {
    system("date");
    system("mkdir arrays4by4-256");

    report_start("calculating index dp");
    // index_dp[cnt][sum] gives the number of ways to get sum in cnt cells (empty allowed)
    // useful for fast indexation inside layers of constant sum; checked to be at most ...
    index_dp[0][0] = 1;
    std::fill(index_dp[0] + 1, index_dp[0] + layers_cnt, 0);
    for (int cnt = 1; cnt <= 16; ++cnt) {
        for (int sum = 0; sum <= 4 * 4 * win_const / 2; ++sum) {
            index_dp[cnt][sum] = index_dp[cnt - 1][sum];
            for (int tile = 2; tile < win_const && tile <= sum; tile *= 2)
                index_dp[cnt][sum] += index_dp[cnt - 1][sum - tile];
        }
    }
    std::cout << "max size of layer: " << *std::max_element(index_dp[16], index_dp[16] + 4 * 4 * win_const / 2 + 1) << "\n";
    report_finish();

    unsigned char *user_turn_positions[layers_cnt], *hater_turn_positions[layers_cnt];  // sized index_dp[16][sum]/8+1
    // values are always _for user_; for example, a unit in hater's array means the position is winnable by user, same as with user's array

    int CPU_concurrency;
    std::cout << "CPU concurrency: ";
    std::cin >> CPU_concurrency;
    std::thread jobs[CPU_concurrency];
    std::cout << "start with layer (default " << 4 * 4 * win_const / 2 << "): ";
    int init_layer_sum;
    std::cin >> init_layer_sum;
    for (int sum = init_layer_sum; sum >= 0; sum -= 2) {
        report_start("processing layer with sum " + itos(sum));
        std::cout << " sizes in bytes of current, current+2, current+4: " << (index_dp[16][sum] / 8 + 1 >> 20) << "M ";
        sum + 2 < layers_cnt ? std::cout << (index_dp[16][sum + 2] / 8 + 1 >> 20) << "M " : std::cout << "[doesn't exist] ";
        sum + 4 < layers_cnt ? std::cout << (index_dp[16][sum + 4] / 8 + 1 >> 20) << "M" : std::cout << "[doesn't exist] ";
        std::cout << std::endl;
        long long total_cnt = index_dp[16][sum];  // in a layer for one player
        if (sum + 2 < layers_cnt) {
            user_turn_positions[sum + 2] = new unsigned char[index_dp[16][sum + 2] / 8 + 1];
            read_arr_from_disk(index_dp[16][sum + 2] / 8 + 1, user_turn_positions[sum + 2], "arrays4by4-256/ulayer" + itos(sum + 2, 4) + ".dat");
        }
        if (sum + 4 < layers_cnt) {
            user_turn_positions[sum + 4] = new unsigned char[index_dp[16][sum + 4] / 8 + 1];
            read_arr_from_disk(index_dp[16][sum + 4] / 8 + 1, user_turn_positions[sum + 4], "arrays4by4-256/ulayer" + itos(sum + 4, 4) + ".dat");
        }
        // first hater positions
        hater_turn_positions[sum] = new unsigned char[index_dp[16][sum] / 8 + 1];
        hater_turn_positions[sum][index_dp[16][sum] / 8] = 0;  // useless line to avoid UB later
        const int kBlockDivision = 14;
        for (int thread_id = 0; thread_id < CPU_concurrency; ++thread_id) {
            jobs[thread_id] = std::thread([&](int thread_id) {
                for (long long block_id = 0; block_id < (total_cnt >> kBlockDivision) + 1; ++block_id)
                    if (hash(block_id) % CPU_concurrency == thread_id)
                        process_hater_layer(user_turn_positions, hater_turn_positions, sum, block_id << kBlockDivision, std::min<long long>(block_id + 1 << kBlockDivision, total_cnt));
            }, thread_id);
        }
        for (int thread_id = 0; thread_id < CPU_concurrency; ++thread_id)
            jobs[thread_id].join();
        if (sum + 2 < layers_cnt)
            delete[] user_turn_positions[sum + 2];
        if (sum + 4 < layers_cnt)
            delete[] user_turn_positions[sum + 4];
        // then user positions
        user_turn_positions[sum] = new unsigned char[index_dp[16][sum] / 8 + 1];
        user_turn_positions[sum][index_dp[16][sum] / 8] = 0;  // useless line to avoid UB later
        for (int thread_id = 0; thread_id < CPU_concurrency; ++thread_id) {
            jobs[thread_id] = std::thread([&](int thread_id) {
                for (long long block_id = 0; block_id < (total_cnt >> kBlockDivision) + 1; ++block_id)
                    if (hash(block_id) % CPU_concurrency == thread_id)
                        process_user_layer(user_turn_positions, hater_turn_positions, sum, block_id << kBlockDivision, std::min<long long>(block_id + 1 << kBlockDivision, total_cnt));
            }, thread_id);
        }
        for (int thread_id = 0; thread_id < CPU_concurrency; ++thread_id)
            jobs[thread_id].join();
        delete[] hater_turn_positions[sum];
        write_arr_to_disk(index_dp[16][sum] / 8 + 1, user_turn_positions[sum], "arrays4by4-256/ulayer" + itos(sum, 4) + ".dat");
        // system("gzip arrays4by4-256/ulayer" + itos(sum, 4) + ".dat &");  -- this is impossible, because it will not allow this program to read this layer later
        delete[] user_turn_positions[sum];
        report_finish();
    }

    std::cout << "FINAL RESULT:\n";
    std::cout << "empty field:\n";
    board empty;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            empty.f[i][j] = 0;
    std::cout << "\tuser " << get_user_value_from_disk(user_turn_positions, empty) << ", ";
    std::cout << "hater " << get_hater_value_from_disk(hater_turn_positions, empty) << "\n";
    std::cout << "field with just one tile:\n";
    for (int player = 0; player < 2; ++player) {
        std::cout << "\t" << (player == 0 ? "user" : "hater") << ":\n";
        for (int i = 0; i < 4; ++i) {
            std::cout << "\t";
            for (int j = 0; j < 4; ++j) {
                board b;
                for (int k = 0; k < 4; ++k)
                    for (int l = 0; l < 4; ++l)
                        b.f[k][l] = k == i && l == j ? 2 : 0;
                if (player == 0)
                    std::cout << get_user_value_from_disk(user_turn_positions, b);
                else
                    std::cout << get_hater_value_from_disk(hater_turn_positions, b);
                b.f[i][j] = 4;
                if (player == 0)
                    std::cout << get_user_value_from_disk(user_turn_positions, b) << " ";
                else
                    std::cout << get_hater_value_from_disk(hater_turn_positions, b) << " ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "field with two tiles:\n";
    for (int player = 0; player < 2; ++player) {
        std::cout << "\n\t" << (player == 0 ? "user" : "hater") << ":\n\n";
        for (int i1 = 0; i1 < 4; ++i1) {
            for (int i2 = 0; i2 < 4; ++i2) {
                std::cout << "\t";
                for (int j1 = 0; j1 < 4; ++j1) {
                    for (int j2 = 0; j2 < 4; ++j2) {
                        if (i1 == i2 && j1 == j2) {
                            std::cout << "---- ";
                            continue;
                        }
                        board b;
                        for (int k = 0; k < 4; ++k)
                            for (int l = 0; l < 4; ++l)
                                b.f[k][l] = 0;
                        for (int u1 = 2; u1 <= 4; u1 += 2) {
                            for (int u2 = 2; u2 <= 4; u2 += 2) {
                                b.f[i1][j1] = u1, b.f[i2][j2] = u2;
                                if (player == 0)
                                    std::cout << get_user_value_from_disk(user_turn_positions, b);
                                else
                                    std::cout << get_hater_value_from_disk(hater_turn_positions, b);
                            }
                        }
                        std::cout << " ";
                    }
                    std::cout << "  ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }


    system("date");

    reenter_pos:

    std::cout << "Enter position:\n";
    board b;
    for (int i = 0; i < 4; ++i) {
        std::cout << "\t";
        for (int j = 0; j < 4; ++j)
            std::cin >> b.f[i][j];
    }
    std::cout << "winnable (for user, i.e. for you): " << get_user_value_from_disk(user_turn_positions, b) << "\n";
    if (get_user_value_from_disk(user_turn_positions, b))
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
        assert(!get_hater_value_from_disk(hater_turn_positions, b));
        std::vector<int> ways;
        for (int i = 0; i < 32; ++i) {
            board test(b);
            if (test.addTile(i) && !get_user_value_from_disk(user_turn_positions, test))
                ways.push_back(i);
        }
        if (ways.empty()) {
            std::cout << "ALGORITHM FAILURE.\n\n";
            goto reenter_pos;
        }
        int move = ways[rand() % ways.size()];
        assert(b.addTile(move) && !get_user_value_from_disk(user_turn_positions, b));
        std::cout << "position after answer: " << b << "\n";
    }
}
