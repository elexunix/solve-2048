#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include "field2byKlib.h"

const int threads = 4096;
static_assert(threads % 256 == 0);
const int allowed_updates_per_thread = 65536;

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
__host__ __device__ bool get_value(unsigned char *arr, const board& b) {
    return get_value(arr, encode_position(b));
}
__host__ __device__ void set_value(unsigned char *arr, const board& b, bool value, long long **updates, int thread_id, long long& thread_iter) {
    set_value(arr, encode_position(b), value, updates, thread_id, thread_iter);
}
__host__ __device__ void really_set_value(unsigned char *arr, const board& b, bool value) {
    really_set_value(arr, encode_position(b), value);
}

const long long log_win_const_to2 = log_win_const * log_win_const;
const long long log_win_const_to4 = log_win_const_to2 * log_win_const_to2;
const long long log_win_const_to6 = log_win_const_to4 * log_win_const_to2;
const long long log_win_const_to8 = log_win_const_to6 * log_win_const_to2;
const long long log_win_const_to10 = log_win_const_to8 * log_win_const_to2;
const long long log_win_const_to12 = log_win_const_to10 * log_win_const_to2;
const long long log_win_const_to14 = log_win_const_to12 * log_win_const_to2;
const long long log_win_const_to16 = log_win_const_to14 * log_win_const_to2;

const long long total_positions = log_win_const_to12;

const long long arr_size = total_positions / 8 + 1;

__global__ void process_layer(unsigned char *arr, int sum, long long **updates, long long *thread_cnts) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x, idy = blockIdx.y * blockDim.y + threadIdx.y;
    // if (idx >= grid_1dim_width || idy >= grid_1dim_width)
    //     return;
    // long long id = idx * grid_1dim_width + idy;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    long long cnt_changes = 0;
    for (long long position_id = thread_id; position_id < total_positions; position_id += gridDim.x * blockDim.x) {
        board position(decode_position(position_id));
        if (position.sum() != sum)
            continue;
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
            for (int shot = 0; shot < 2 * H * W; ++shot) {
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
    for (int sum = H * W * win_const / 2; sum >= 0; sum -= 2) {
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
    for (int i = 0; i < H; ++i) {
        std::cout << '\t';
        for (int j = 0; j < W; ++j) {
            board b;
            for (int k = 0; k < H; ++k)
                for (int l = 0; l < W; ++l)
                    b.f[k][l] = k == i && l == j ? 2 : 0;
            std::cout << get_value(arr, encode_position(b));
            for (int k = 0; k < H; ++k)
                for (int l = 0; l < W; ++l)
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
    for (int i = 0; i < H; ++i) {
        std::cout << '\t';
        for (int j = 0; j < W; ++j)
            std::cin >> b.f[i][j];
    }


    std::cout << "initial code: " << encode_position(b) << '\n';
    if (get_value(arr, b)) {
        std::cout << "this position is winning...\n";
        while (true) {
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
                    std::cout << "answer swipe: " << swipe << '\n';
                    b = temp;
                    break;
                }
                bool excellent = true;
                for (int shot = 0; shot < 2 * H * W; ++shot) {
                    board test(temp);
                    if (test.addTile(shot))
                        std::cout << "[swipe " << swipe << ", shot " << shot << ", value " << get_value(arr, test) << "]\n",
                        excellent &= get_value(arr, test);
                }
                if (excellent) {
                    std::cout << "answer swipe: " << swipe << '\n';
                    b = temp;
                    break;
                }
            }
            std::cout << "now position (" << encode_position(b) << "):\n";
            b.print();
            std::cout << "value: " << get_value(arr, b) << '\n';
            std::cout << "your new tile: ";
            reenter_tile:
            int i, j, tile;
            std::cin >> i >> j >> tile;
            if (i == 42)
                goto reenter_pos;
            if (i < 0 || i >= H|| j < 0 || j >= W || tile != 2 && tile != 4 || b.f[i][j] != 0) {
                std::cout << "invalid, please reenter;\n";
                goto reenter_tile;
            }
            b.f[i][j] = tile;
            std::cout << "new position: ";
            b.print();


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
                    std::cout << "out position " << encode_position(b) << " is already won\n";
                    continue;
                }
                bool excellent = true;
                for (int shot = 0; shot < 2 * H * W; ++shot) {
                    board test(temp);
                    if (test.addTile(shot))
                        std::cout << "[swipe " << swipe << ", shot " << shot << ", value " << get_value(arr, test) << "]\n",
                        excellent &= get_value(arr, test);
                }
                winnable |= excellent;
            }
            std::cout << "winnable according to the rule: " << winnable << std::endl;
        }
    }
    else {
        std::cout << "this position is losing...\n";
        while (true) {
            std::cout << "your swipe: ";
            std::string swipe;
            reenter_swipe:
            std::cin >> swipe;
            if (swipe == "left" || swipe == "Left" || swipe == "LEFT")
                b.swipeLeft();
            else if (swipe == "right" || swipe == "Right" || swipe == "RIGHT")
                b.swipeRight();
            else if (swipe == "up" || swipe == "Up" || swipe == "UP")
                b.swipeUp();
            else if (swipe == "down" || swipe == "Down" || swipe == "DOWN")
                b.swipeDown();
            else if (swipe == "reenter")
                goto reenter_pos;
            else {
                std::cout << "sorry, wasn't understood, please reenter\n";
                goto reenter_swipe;
            }
            std::cout << "position (" << encode_position(b) << ") after swipe:\n";
            b.print();
            for (int shot = 0; shot < 2 * H * W; ++shot) {
                board test(b);
                if (test.addTile(shot) && !get_value(arr, test)) {
                    std::cout << "move " << shot << " is the answer\n";
                    b = test;
                    break;
                }
            }
            std::cout << "now position (" << encode_position(b) << "):\n";
            b.print();
            std::cout << "its value: " << get_value(arr, b) << '\n';


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
                for (int shot = 0; shot < 2 * H * W; ++shot) {
                    board test(temp);
                    if (test.addTile(shot))
                        excellent &= get_value(arr, test);
                }
                winnable |= excellent;
            }
            std::cout << "winnable according to the rule: " << winnable << std::endl;
        }
    }

    return 0;
}