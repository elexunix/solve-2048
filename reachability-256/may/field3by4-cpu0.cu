#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include "field3by4lib0.h"

const long long arr_size = log_win_const * log_win_const * log_win_const * log_win_const * log_win_const * log_win_const
                * (long long)log_win_const * log_win_const * log_win_const * log_win_const * log_win_const * log_win_const / 8 + 1;

__host__ __device__ bool get_value(unsigned char *arr, long long pos) {
    return arr[pos >> 3] & 1 << (pos & 7);
}
__host__ __device__ void set_value(unsigned char *arr, long long pos, bool value) {
    if (value != get_value(arr, pos))
        arr[pos >> 3] ^= 1 << (pos & 7);
}

__host__ __device__ int encode_inside_layer(const board& b, int sum, const int dp[13][3 * 4 * win_const / 2 + 1]) {
    int result = 0, left = 11;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j, --left) {
            if (b.f[i][j] == 0)
                continue;
            result += dp[left][sum];
            for (int tile = 2; tile < b.f[i][j]; tile *= 2)
                result += dp[left][sum - tile];
            sum -= b.f[i][j];
        }
    }
    // assert(sum == 0);
    return result;
}

__host__ __device__ board decode_inside_layer(int id, int sum, const int dp[13][3 * 4 * win_const / 2 + 1]) {
    board result;
    int left = 11;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j, --left) {
            if (id < dp[left][sum]) {
                result.f[i][j] = 0;
                continue;
            }
            id -= dp[left][sum];
            int tile = 2;
            while (id >= dp[left][sum - tile])
                id -= dp[left][sum - tile], tile *= 2;
            result.f[i][j] = tile;
            sum -= tile;
        }
    }
    // assert(sum == 0);
    return result;
}

int index_dp[13][3 * 4 * win_const / 2 + 1];

const int CPU_concurrency = 1;

void process_layer(unsigned char *arr, int sum, int thread_id) {
    for (int position_id = thread_id; position_id < index_dp[12][sum]; position_id += CPU_concurrency) {
        board position(decode_inside_layer(position_id, sum, index_dp));
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
                if (test.addTile(shot))
                    excellent &= get_value(arr, encode_position(test));
            }
            winnable |= excellent;
        }
        set_value(arr, position_id, winnable);
    }
}

int main() {
    system("date");
    unsigned char *arr;
    report_start("allocating global array (" + itos(arr_size) + ")");
    arr = new unsigned char[arr_size];
    report_finish();

    report_start("calculating index dp");
    // index_dp[cnt][sum] gives the number of ways to get sum using cnt tiles (empty included)
    // useful for fast indexation inside layers of constant sum; checked to be at most 150246756
    index_dp[0][0] = 1;
    std::fill(index_dp[0] + 1, index_dp[0] + 3 * 4 * win_const / 2 + 1, 0);
    for (int cnt = 1; cnt <= 12; ++cnt) {
        for (int sum = 0; sum <= 3 * 4 * win_const / 2; ++sum) {
            index_dp[cnt][sum] = index_dp[cnt - 1][sum];
            for (int tile = 2; tile < win_const && tile <= sum; tile *= 2)
                index_dp[cnt][sum] += index_dp[cnt - 1][sum - tile];
            assert(index_dp[cnt][sum] >= index_dp[cnt - 1][sum]);
        }
    }
    report_finish();

    for (int sum = 3 * 4 * win_const / 2; sum >= 0; --sum) {
        if (index_dp[12][sum] == 0)
            continue;
        report_start("processing layer with sum " + itos(sum));
        process_layer(arr, sum, 0);
        report_finish();
    }


    std::cout << "FINAL RESULT:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << '\t';
        for (int j = 0; j < 4; ++j) {
            board b;
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 4; ++l)
                    b.f[k][l] = k == i && l == j ? 2 : 0;
            std::cout << get_value(arr, encode_position(b));
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 4; ++l)
                    b.f[k][l] = k == i && l == j ? 4 : 0;
            std::cout << get_value(arr, encode_position(b)) << ' ';
        }
        std::cout << '\n';
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
    std::cout << "initial code: " << encode_position(b) << '\n';
    if (get_value(arr, encode_position(b))) {
        std::cout << "this position is winning... enter another one\n";
        goto reenter_pos;
    }

    while (true) {
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
        else
            goto reenter_pos;
        if (!possible)
            goto reenter_pos;
        std::cout << "position (" << encode_position(b) << "," << b.sum() << ","
                << encode_inside_layer(b, b.sum(), index_dp) << ") after swipe:\n";
        b.print();
        for (int i = 0; i < 24; ++i) {
            board test(b);
            if (test.addTile(i) && !get_value(arr, encode_position(test))) {
                b = test;
                break;
            }
        }
        std::cout << "now position (" << encode_position(b) << "," << b.sum() << ","
                << encode_inside_layer(b, b.sum(), index_dp) << "):\n";
        b.print();
        std::cout << "its value: " << get_value(arr, encode_position(b)) << '\n';

        bool winnable = false;
        for (int swipe = 0; swipe < 4; ++swipe) {
            board temp(b);
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
                std::cout << "out position " << encode_position(b) << " is almost won\n";
                continue;
            }
            bool excellent = true;
            for (int shot = 0; shot < 24; ++shot) {
                board test(temp);
                if (test.addTile(shot))
                    excellent &= get_value(arr, encode_position(test));
            }
            winnable |= excellent;
        }
        std::cout << "winnable according to the rule: " << winnable << std::endl;
    }
}