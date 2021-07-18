#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include "field3by4lib.h"

// takes 15 min to solve, cf. with previous (2h 12 min)

const long long arr_size = log_win_const * log_win_const * log_win_const * log_win_const * log_win_const * log_win_const
                * (long long)log_win_const * log_win_const * log_win_const * log_win_const * log_win_const * log_win_const / 8 + 1;

const int threads = 4096 * 4;
static_assert(threads % 1024 == 0);
const int allowed_updates_per_thread = 12000;

__host__ __device__ bool get_value(unsigned char *arr, long long pos) {
    return arr[pos >> 3] & 1 << (pos & 7);
}
__host__ __device__ void set_value(unsigned char *arr, long long pos, bool value, long long **updates, int thread_id, long long& thread_iter) {
    updates[thread_id][thread_iter++] = pos | (long long)value << 40;
}
__host__ __device__ void really_set_value(unsigned char *arr, long long pos, bool value) {
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

__global__ void process_layer(unsigned char *arr, int sum, long long **updates, long long *thread_cnts, const int **global_dp) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int local_dp[13][3 * 4 * win_const / 2 + 1];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 13; ++i)
            for (int j = 0; j < 3 * 4 * win_const / 2 + 1; ++j)
                local_dp[i][j] = global_dp[i][j];
    }
    __syncthreads();
    long long cnt_changes = 0;
    for (int position_id = thread_id; position_id < local_dp[12][sum]; position_id += gridDim.x * blockDim.x) {
        board position(decode_inside_layer(position_id, sum, local_dp));
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
        if (cnt_changes < allowed_updates_per_thread)
            set_value(arr, position_id, winnable, updates, thread_id, cnt_changes);
    }
    thread_cnts[thread_id] = cnt_changes;
}

int main() {
    system("date");
    unsigned char *arr;
    report_start("allocating global array (" + itos(arr_size) + ")");
    gpuErrchk(cudaMallocManaged(&arr, arr_size));
    report_finish();

    report_start("allocating update arrays (" + itos(threads * (allowed_updates_per_thread * sizeof(long long) + 2 * sizeof(long long*))) + ")");
    long long **updates, *cnt_updates;
    cudaMallocManaged(&updates, threads * sizeof(long long*));
    for (int i = 0; i < threads; ++i)
        cudaMallocManaged(&updates[i], allowed_updates_per_thread * sizeof(long long));
    cudaMallocManaged(&cnt_updates, threads * sizeof(long long*));
    report_finish();

    report_start("calculating dp");
    int **dp;
    gpuErrchk(cudaMallocManaged(&dp, 13 * sizeof(int*)));
    for (int i = 0; i < 13; ++i)
        gpuErrchk(cudaMallocManaged(&dp[i], (3 * 4 * win_const / 2 + 1) * sizeof(int)));
    // dp[cnt][sum] gives the number of ways to get sum using cnt tiles (empty included)
    // useful for fast indexation inside layers of constant sum; checked to be at most 150246756
    dp[0][0] = 1;
    std::fill(dp[0] + 1, dp[0] + 3 * 4 * win_const / 2 + 1, 0);
    for (int cnt = 1; cnt <= 12; ++cnt) {
        for (int sum = 0; sum <= 3 * 4 * win_const / 2; ++sum) {
            dp[cnt][sum] = dp[cnt - 1][sum];
            for (int tile = 2; tile < win_const && tile <= sum; tile *= 2)
                dp[cnt][sum] += dp[cnt - 1][sum - tile];
            assert(dp[cnt][sum] >= dp[cnt - 1][sum]);
        }
    }
    int dp_like_arr[13][3 * 4 * win_const / 2 + 1];
    for (int i = 0; i < 13; ++i)
        std::copy(dp[i], dp[i] + 3 * 4 * win_const / 2 + 1, dp_like_arr[i]);
    report_finish();

    int global_max_updates = 0;
    for (int sum = 3 * 4 * win_const / 2; sum >= 0; --sum) {
        if (dp[12][sum] == 0)
            continue;
        report_start("processing layer with sum " + itos(sum));
        process_layer<<<threads/1024,1024>>>(arr, sum, updates, cnt_updates, const_cast<const int**>(dp));
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "[max changes cnt is " << *std::max_element(cnt_updates, cnt_updates + threads) << "]";
        global_max_updates = std::max<int>(global_max_updates, *std::max_element(cnt_updates, cnt_updates + threads));
        report_finish();
        report_start("performing changes");
        for (int i = 0; i < threads; ++i)
            for (int j = 0; j < cnt_updates[i]; ++j)
                really_set_value(arr, updates[i][j] & (1ll << 40) - 1, updates[i][j] >> 40 & 1);
        report_finish();
    }
    std::cout << global_max_updates << " is the global max updates cnt\n";


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
                << encode_inside_layer(b, b.sum(), dp_like_arr) << ") after swipe:\n";
        b.print();
        for (int i = 0; i < 24; ++i) {
            board test(b);
            if (test.addTile(i) && !get_value(arr, encode_position(test))) {
                b = test;
                break;
            }
        }
        std::cout << "now position (" << encode_position(b) << "," << b.sum() << ","
                << encode_inside_layer(b, b.sum(), dp_like_arr) << "):\n";
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