#include <algorithm>
#include <bits/stdc++.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <ostream>
#include <random>
#include <thread>
#include <vector>
#include "field4by4lib-cpu.h"


long long estimate_user_layer(int sum, long long cnt_samples, double p_h_sum) {
    long long winnable_for_user = 0;
    std::mt19937 generator(time(0));
    long long layer_size = index_dp[16][sum], layer_size_2 = index_dp[16][sum + 2], layer_size_4 = index_dp[16][sum + 4];
    for (long long i = 0; i < cnt_samples; ++i) {
        board position(decode_inside_layer((((int64_t)generator()<<30)+generator()) % layer_size, sum));
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
            winnable |= possible && (temp.won() || (((int64_t)generator()<<30)+generator()) % layer_size < p_h_sum * layer_size);
        }
        winnable_for_user += winnable;
    }
    return winnable_for_user;
}
long long estimate_hater_layer(int sum, long long cnt_samples, double p_sum_plus_2, double p_sum_plus_4) {
    long long winnable_for_user = 0;
    std::mt19937 generator(time(0));
    long long layer_size = index_dp[16][sum], layer_size_2 = index_dp[16][sum + 2], layer_size_4 = index_dp[16][sum + 4];
    for (long long i = 0; i < cnt_samples; ++i) {
        board position(decode_inside_layer((((int64_t)generator()<<30)+generator()) % layer_size, sum));
        bool winnable = true;
        for (int shot = 0; shot < 32; ++shot) {
            board temp(position);
            winnable &= !temp.addTile(shot) || temp.won() || (((int64_t)generator()<<30)+generator()) % (shot&1?layer_size_2:layer_size_4) < (shot&1?p_sum_plus_2:p_sum_plus_4) * (shot&1?layer_size_2:layer_size_4);
        }
        winnable_for_user += winnable;
    }
    return winnable_for_user;
}


int main(int argc, char **argv) {
    fill_index_dp();
    /*int sum;
    std::cout << "sum: ";
    std::cin >> sum;
    double p_sum_plus_2, p_sum_plus_4;
    std::cout << "p(" << sum + 2 << "): ";
    std::cin >> p_sum_plus_2;
    std::cout << "p(" << sum + 4 << "): ";
    std::cin >> p_sum_plus_4;
    int cnt_samples;
    std::cout << "cnt samples: ";
    std::cin >> cnt_samples;
    std::cout << "hater layer...\t", std::cout.flush();
    double p_h_sum = (double)estimate_hater_layer(sum, cnt_samples, p_sum_plus_2, p_sum_plus_4) / cnt_samples;
    std::cout << p_h_sum << "\n";
    std::cout << "user layer...\t", std::cout.flush();
    double p_sum = (double)estimate_user_layer(sum, cnt_samples, p_h_sum) / cnt_samples;
    std::cout << p_sum << "\n";*/
    std::stringstream ss_start_sum(argv[1]), ss_p_sum_plus_2(argv[2]), ss_p_sum_plus_4(argv[3]), ss_cnt_samples(argv[4]), ss_prediction_len(argv[5]);
    int start_sum, cnt_samples, prediction_len;
    double p_sum_plus_2, p_sum_plus_4;
    ss_start_sum >> start_sum;
    ss_p_sum_plus_2 >> p_sum_plus_2;
    ss_p_sum_plus_4 >> p_sum_plus_4;
    ss_cnt_samples >> cnt_samples;
    ss_prediction_len >> prediction_len;
    assert(start_sum > 0 && start_sum <= 2044);
    assert(0 <= p_sum_plus_2 && p_sum_plus_2 <= 1);
    assert(0 <= p_sum_plus_4 && p_sum_plus_4 <= 1);
    assert(cnt_samples > 0);
    assert(0 < prediction_len && 2 * prediction_len <= start_sum);
    for (int i = 0, sum = start_sum; i < prediction_len; ++i, sum -= 2) {
        double p_h_sum = (double)estimate_hater_layer(sum, cnt_samples, p_sum_plus_2, p_sum_plus_4) / cnt_samples;
        double p_sum = (double)estimate_user_layer(sum, cnt_samples, p_h_sum) / cnt_samples;
        std::cout << p_sum << std::endl;
        p_sum_plus_4 = p_sum_plus_2, p_sum_plus_2 = p_sum;
    }
}
