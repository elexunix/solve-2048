#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include "field3by4lib-anti.h"

const long long arr_size = log_win_const * log_win_const * log_win_const * log_win_const * log_win_const * log_win_const
                    * (long long)log_win_const * log_win_const * log_win_const * log_win_const * log_win_const * log_win_const / 8 + 1;

const int threads = 4096;
static_assert(threads % 256 == 0);
const int allowed_updates_per_thread = 65536;

// this causes race conditions, by the way
__host__ __device__ bool get_value(unsigned char *arr, long long pos) {
    // return arr[pos];
    return arr[pos >> 3] & 1 << (pos & 7);
}

__host__ __device__ void set_value(unsigned char *arr, long long pos, bool value, long long **updates, int thread_id, long long& thread_iter) {
    // arr[pos] = value;
    // if (value != get_value(arr, pos))
    //     arr[pos >> 3] ^= 1 << (pos & 7);
    updates[thread_id][thread_iter++] = pos | (long long)value << 40;
}

__host__ __device__ void really_set_value(unsigned char *arr, long long pos, bool value) {
    // arr[pos] = value;
    if (value != get_value(arr, pos))
        arr[pos >> 3] ^= 1 << (pos & 7);
}

// const long long grid_1dim_width = log_win_const * log_win_const * log_win_const * log_win_const * log_win_const * log_win_const;
// const dim3 dimGrid(grid_1dim_width / 32 + 1, grid_1dim_width / 32 + 1), dimBlock(32, 32);
const long long log_win_const_to6 = log_win_const * log_win_const * log_win_const * log_win_const * log_win_const * log_win_const;
const long long log_win_const_to12 = log_win_const_to6 * log_win_const_to6;

__global__ void process_layer(unsigned char *arr, int sum, long long **updates, long long *thread_cnts) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x, idy = blockIdx.y * blockDim.y + threadIdx.y;
    // if (idx >= grid_1dim_width || idy >= grid_1dim_width)
    //     return;
    // long long id = idx * grid_1dim_width + idy;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    long long cnt_changes = 0;
    for (long long position_id = thread_id; position_id < log_win_const_to12; position_id += gridDim.x * blockDim.x) {
        board position(decode_position(position_id));
        if (position.sum() != sum)
            continue;
        bool winnable = false, some_possible = false;
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
            some_possible |= possible;
            if (temp.won())
                continue;
            if (temp.lost()) {
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
            set_value(arr, position_id, winnable || !some_possible, updates, thread_id, cnt_changes);
    }
    thread_cnts[thread_id] = cnt_changes;
}

int main() {
    unsigned char *arr;
    report_start("allocating global array (" + itos(arr_size) + ")");
    gpuErrchk(cudaMallocManaged(&arr, arr_size));
    report_finish();

    // report_start("filling global array with zeros");
    // fill_zeros<<<dimGrid, dimBlock>>>(arr);
    // cudaDeviceSynchronize();
    // report_finish();

    report_start("allocating update arrays (" + itos(threads * (allowed_updates_per_thread * sizeof(long long) + 2 * sizeof(long long*))) + ")");
    long long **updates, *cnt_updates;
    cudaMallocManaged(&updates, threads * sizeof(long long*));
    for (int i = 0; i < threads; ++i)
        cudaMallocManaged(&updates[i], allowed_updates_per_thread * sizeof(long long));
    cudaMallocManaged(&cnt_updates, threads * sizeof(long long*));
    report_finish();

    int global_max_updates = 0;
    for (int sum = 3 * 4 * win_const / 2; sum >= 0; sum -= 2) {
        report_start("processing layer with sum " + itos(sum));
        process_layer<<<threads/256,256>>>(arr, sum, updates, cnt_updates);
        gpuErrchk(cudaDeviceSynchronize());
        /*std::cout << "changes cnt by threads:";
        for (int i = 0; i < threads; ++i)
            std::cout << ' ' << cnt_updates[i];*/
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

    // std::cout << "Discussed position (97734630):\n";
    // decode_position(97734630).print();
    /*while (true) {
        std::cout << "\n\nEnter position:\n";
        board b;
        for (int i = 0; i < 3; ++i) {
            std::cout << '\t';
            for (int j = 0; j < 4; ++j)
                std::cin >> b.f[i][j];
        }
        std::cout << "code: " << encode_position(b) << '\n';
        std::cout << "decoding for check:\n";
        decode_position(encode_position(b)).print();
        std::cout << "is won: " << b.won() << '\n';
        std::cout << "is lost: " << b.lost() << '\n';
        std::cout << "is finished: " << b.finished() << '\n';
        std::cout << "value: " << get_value(arr, encode_position(b)) << '\n';
    }*/
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
    assert(get_value(arr, encode_position(b)) == 0);
    while (true) {
        std::cout << "your swipe: ";
        int swipe;
        std::cin >> swipe;
        switch (swipe) {
            case 0: b.swipeLeft(); break;
            case 1: b.swipeRight(); break;
            case 2: b.swipeUp(); break;
            case 3: b.swipeDown(); break;
            case 4: goto reenter_pos;
        }
        std::cout << "position (" << encode_position(b) << ") after swipe:\n";
        b.print();
        for (int i = 0; i < 24; ++i) {
            board test(b);
            if (test.addTile(i) && !get_value(arr, encode_position(test))) {
                std::cout << "move " << i << " is the answer\n";
                b = test;
                break;
            }
        }
        std::cout << "now position (" << encode_position(b) << "):\n";
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

    report_start("freeing memory");
    cudaFree(arr);
    report_finish();
    return 0;
}