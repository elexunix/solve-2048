#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include "field2by2by2lib.h"

const long long log_win_const_to4 = log_win_const * log_win_const * log_win_const * log_win_const;
const long long log_win_const_to8 = log_win_const_to4 * log_win_const_to4;
const long long arr_size = log_win_const_to8 / 8 + 1;

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

__global__ void process_layer(unsigned char *arr, int sum, long long **updates, long long *thread_cnts) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    long long cnt_changes = 0;
    for (long long position_id = thread_id; position_id < log_win_const_to8; position_id += gridDim.x * blockDim.x) {
        board position(decode_position(position_id));
        if (position.sum() != sum)
            continue;
        bool winnable = false;
        for (int swipe = 0; swipe < 6; ++swipe) {
            board temp(position);
            bool possible;
            switch (swipe) {
                case 0: possible = temp.swipeLeft(); break;
                case 1: possible = temp.swipeRight(); break;
                case 2: possible = temp.swipeUp(); break;
                case 3: possible = temp.swipeDown(); break;
                case 4: possible = temp.swipeFront(); break;
                case 5: possible = temp.swipeBack(); break;
            }
            if (!possible)
                continue;
            if (temp.won()) {
                winnable = true;
                break;
            }
            bool excellent = true;
            for (int shot = 0; shot < 16; ++shot) {
                board test(temp);
                if (test.addTile(shot))
                    excellent &= get_value(arr, test);
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

    report_start("allocating update arrays (" + itos(threads * (allowed_updates_per_thread * sizeof(long long) + 2 * sizeof(long long*))) + ")");
    long long **updates, *cnt_updates;
    cudaMallocManaged(&updates, threads * sizeof(long long*));
    for (int i = 0; i < threads; ++i)
        cudaMallocManaged(&updates[i], allowed_updates_per_thread * sizeof(long long));
    cudaMallocManaged(&cnt_updates, threads * sizeof(long long*));
    report_finish();

    int global_max_updates = 0;
    for (int sum = 2 * 2 * 2 * win_const / 2; sum >= 0; sum -= 2) {
        report_start("processing layer with sum " + itos(sum));
        process_layer<<<threads/256,256>>>(arr, sum, updates, cnt_updates);
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
    for (int j = 0; j < 2; ++j) {
        std::cout << '\t';
        for (int i = 0; i < 2; ++i) {
            board b;
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 2; ++l)
                    for (int m = 0; m < 2; ++m)
                        for (int n = 0; n < 2; ++n)
                            b.f[l][m][n] = l == i && m == j && n == k ? 2 : 0;
                std::cout << get_value(arr, b);
                for (int l = 0; l < 2; ++l)
                    for (int m = 0; m < 2; ++m)
                        for (int n = 0; n < 2; ++n)
                            b.f[l][m][n] = l == i && m == j && n == k ? 4 : 0;
                std::cout << get_value(arr, b) << ' ';
            }
            std::cout << "\t\t";
        }
        std::cout << '\n';
    }

    reenter_pos:
    std::cout << "Enter position:\n";
    board b;
    for (int i = 0; i < 2; ++i) {
        std::cout << "\tlayer " << i << ":\n";
        for (int j = 0; j < 2; ++j) {
            std::cout << '\t';
            for (int k = 0; k < 2; ++k)
                std::cin >> b.f[i][j][k];
        }
    }
    std::cout << "initial code: " << encode_position(b) << '\n';
    if (get_value(arr, b)) {
        std::cout << "this position is winning...\n";
        while (true) {
            for (int swipe = 0; swipe < 6; ++swipe) {
                board temp(b);
                bool possible;
                switch (swipe) {
                    case 0: possible = temp.swipeLeft(); break;
                    case 1: possible = temp.swipeRight(); break;
                    case 2: possible = temp.swipeUp(); break;
                    case 3: possible = temp.swipeDown(); break;
                    case 4: possible = temp.swipeFront(); break;
                    case 5: possible = temp.swipeBack(); break;
                }
                if (!possible)
                    continue;
                if (temp.won()) {
                    std::cout << "answer swipe: " << swipe << '\n';
                    b = temp;
                    break;
                }
                bool excellent = true;
                for (int shot = 0; shot < 16; ++shot) {
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
            int i, j, k, tile;
            std::cin >> i >> j >> k >> tile;
            if (i == 42)
                goto reenter_pos;
            if (i < 0 || i > 1 || j < 0 || j > 1 || k < 0 || k > 1 || tile != 2 && tile != 4 || b.f[i][j][k] != 0) {
                std::cout << "invalid, please reenter;\n";
                goto reenter_tile;
            }
            b.f[i][j][k] = tile;
            std::cout << "new position: ";
            b.print();


            bool winnable = false;
            for (int swipe = 0; swipe < 6; ++swipe) {
                board temp(b);
                bool possible;
                switch (swipe) {
                    case 0: possible = temp.swipeLeft(); break;
                    case 1: possible = temp.swipeRight(); break;
                    case 2: possible = temp.swipeUp(); break;
                    case 3: possible = temp.swipeDown(); break;
                    case 4: possible = temp.swipeFront(); break;
                    case 5: possible = temp.swipeBack(); break;
                }
                if (!possible)
                    continue;
                if (temp.won()) {
                    std::cout << "out position " << encode_position(b) << " is already won\n";
                    continue;
                }
                bool excellent = true;
                for (int shot = 0; shot < 16; ++shot) {
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
            else if (swipe == "front" || swipe == "Front" || swipe == "FRONT")
                b.swipeFront();
            else if (swipe == "back" || swipe == "Back" || swipe == "BACK")
                b.swipeBack();
            else if (swipe == "reenter")
                goto reenter_pos;
            else {
                std::cout << "sorry, wasn't understood, please reenter\n";
                goto reenter_swipe;
            }
            std::cout << "position (" << encode_position(b) << ") after swipe:\n";
            b.print();
            for (int shot = 0; shot < 16; ++shot) {
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
            for (int swipe = 0; swipe < 6; ++swipe) {
                board temp(b);
                bool possible;
                switch (swipe) {
                    case 0: possible = temp.swipeLeft(); break;
                    case 1: possible = temp.swipeRight(); break;
                    case 2: possible = temp.swipeUp(); break;
                    case 3: possible = temp.swipeDown(); break;
                    case 4: possible = temp.swipeFront(); break;
                    case 5: possible = temp.swipeBack(); break;
                }
                if (!possible)
                    continue;
                if (temp.won()) {
                    std::cout << "out position " << encode_position(b) << " is almost won\n";
                    continue;
                }
                bool excellent = true;
                for (int shot = 0; shot < 16; ++shot) {
                    board test(temp);
                    if (test.addTile(shot))
                        excellent &= get_value(arr, test);
                }
                winnable |= excellent;
            }
            std::cout << "winnable according to the rule: " << winnable << std::endl;
        }
    }

    // freeing not performed, as this part is not accessible
    return 0;
}