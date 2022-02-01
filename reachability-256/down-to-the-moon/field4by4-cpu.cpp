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
            //assert(temp.sum() == sum);
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

    fill_index_dp();

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
        system("date");
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

    report_final_result(user_turn_positions, hater_turn_positions);

    system("date");
}
