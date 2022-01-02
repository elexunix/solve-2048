#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include "field3by5lib-cpu.h"

// try to rewrite set_value, check performance
// try to rewrite operator[]s, check performance


bool get_value(const unsigned char *arr, long long pos) {
    return arr[pos >> 3] & 1 << (pos & 7);
}
void set_value(unsigned char *arr, long long pos, bool value) {
    if (value != get_value(arr, pos))
        arr[pos >> 3] ^= 1 << (pos & 7);
}


const int layers_cnt = 3 * 5 * win_const / 2 + 1;
long long index_dp[16][layers_cnt];

long long encode_inside_layer(const board &b, int sum) {
    // assert(b.sum() == sum);
    long long result = 0;
    int left = 14;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j, --left) {
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
    int left = 14;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j, --left) {
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

bool get_value(const unsigned char *const arr[layers_cnt], const board &b, int hint_sum) {
    return get_value(arr[hint_sum], encode_inside_layer(b, hint_sum));
}
bool get_value(const unsigned char *const arr[layers_cnt], const board &b) {
    return get_value(arr, b, b.sum());
}


// const int CPU_concurrency = 5;
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
        for (int shot = 0; shot < 30; ++shot) {
            board temp(position);
            winnable &= !temp.addTile(shot) || get_value(user_layers, temp);
        }
        set_value(hater_layers[sum], position_id, winnable);
    }
}


void write_arr_to_disk(long long n, const unsigned char *arr, const std::string filename) {
    // report_start("writing to \"" + filename + "\"");
    std::ofstream fout(filename, std::ios::out | std::ios::binary);
    fout.write(reinterpret_cast<const char*>(arr), n);
    // report_finish();
}
void read_arr_from_disk(long long n, unsigned char *arr, const std::string filename) {
    // report_start("reading \"" + filename + "\"");
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    if (!fin.is_open())
        std::cout << "UNABLE TO OPEN " << filename << "\n";
    fin.read(reinterpret_cast<char*>(arr), n);
    // report_finish();
}

bool get_user_value_from_disk_slow(unsigned char *arr[layers_cnt], const board &b) [[deprecated]] {
    arr[b.sum()] = new unsigned char[index_dp[15][b.sum()] / 8 + 1];
    read_arr_from_disk(index_dp[15][b.sum()] / 8 + 1, arr[b.sum()], "arrays3by5/ulayer" + itos(b.sum(), 4) + ".dat");
    bool result = get_value(arr, b);
    delete[] arr[b.sum()];
    return result;
}
bool get_user_value_from_disk(unsigned char *arr[layers_cnt], const board &b) {
    return get_user_value_from_disk_slow(arr, b);
}
/*bool get_hater_value_from_disk(unsigned char *arr[layers_cnt], const board &b) [[deprecated]] {  // impossible to provide unless recorded
    arr[b.sum()] = new unsigned char[index_dp[15][b.sum()] / 8 + 1];
    read_arr_from_disk(index_dp[15][b.sum()] / 8 + 1, arr[b.sum()], "arrays3by5/hlayer" + itos(b.sum(), 4) + ".dat");
    bool result = get_value(arr, b);
    delete[] arr[b.sum()];
    return result;
}*/
bool get_hater_value_from_disk_damn_slow(unsigned char *arr[layers_cnt], const board &b) {
    for (int way = 0; way < 30; ++way) {
        board temp(b);
        if (temp.addTile(way) && !get_user_value_from_disk(arr, temp))
            return false;
    }
    return true;
}
bool get_hater_value_from_disk(unsigned char *arr[layers_cnt], const board &b) {
    return get_hater_value_from_disk_damn_slow(arr, b);
}


void system(const std::string command) {
    system(command.c_str());
}


int main() {
    // const long long sz = 8'000'000'000;
    // unsigned char *t = new unsigned char[sz];
    // t[0] = 42;
    // for (long long i = 1; i < sz; ++i)
    //     t[i] = t[i - 1] * t[i - 1] + 1;
    // std::cout << std::accumulate(t, t + sz, 0);
    system("date");
    system("mkdir arrays3by5-128");

    report_start("calculating index dp");
    // index_dp[cnt][sum] gives the number of ways to get sum in cnt cells (empty allowed)
    // useful for fast indexation inside layers of constant sum; checked to be at most ...
    index_dp[0][0] = 1;
    std::fill(index_dp[0] + 1, index_dp[0] + layers_cnt, 0);
    for (int cnt = 1; cnt <= 15; ++cnt) {
        for (int sum = 0; sum <= 3 * 5 * win_const / 2; ++sum) {
            index_dp[cnt][sum] = index_dp[cnt - 1][sum];
            for (int tile = 2; tile < win_const && tile <= sum; tile *= 2)
                index_dp[cnt][sum] += index_dp[cnt - 1][sum - tile];
        }
    }
    std::cout << "max size of layer: " << *std::max_element(index_dp[15], index_dp[15] + 3 * 5 * win_const / 2 + 1) << "\n";
    report_finish();

    unsigned char *user_turn_positions[layers_cnt], *hater_turn_positions[layers_cnt];  // sized index_dp[15][sum]/8+1
    // values are always _for user_; for example, a unit in hater's array means the position is winnable by user, same as with user's array

    std::cout << "FINAL RESULT:\n";
    std::cout << "empty field:\n";
    board empty;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            empty.f[i][j] = 0;
    std::cout << "\tuser " << get_user_value_from_disk(user_turn_positions, empty) << ", ";
    std::cout << "hater " << get_hater_value_from_disk(hater_turn_positions, empty) << "\n";
    std::cout << "field with just one tile:\n";
    for (int player = 0; player < 2; ++player) {
        std::cout << "\t" << (player == 0 ? "user" : "hater") << ":\n";
        for (int i = 0; i < 3; ++i) {
            std::cout << "\t";
            for (int j = 0; j < 5; ++j) {
                board b;
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 5; ++l)
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
        for (int i1 = 0; i1 < 3; ++i1) {
            for (int i2 = 0; i2 < 3; ++i2) {
                std::cout << "\t";
                for (int j1 = 0; j1 < 5; ++j1) {
                    for (int j2 = 0; j2 < 5; ++j2) {
                        if (i1 == i2 && j1 == j2) {
                            std::cout << "---- ";
                            continue;
                        }
                        board b;
                        for (int k = 0; k < 3; ++k)
                            for (int l = 0; l < 5; ++l)
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
    for (int i = 0; i < 3; ++i) {
        std::cout << '\t';
        for (int j = 0; j < 5; ++j)
            std::cin >> b.f[i][j];
    }
    std::cout << "this user position is winnable for user: " << get_user_value_from_disk(user_turn_positions, b) << "\n";
    std::cout << "this hater position is winnable for user: " << get_hater_value_from_disk(hater_turn_positions, b) << "\n";
    if (!get_hater_value_from_disk(hater_turn_positions, b))
        goto reenter_pos;

    while (true) {
        reenter_tile:
        int i_coord, j_coord, tile;
        std::cout << "your tile i coordinate (0-2): ";
        std::cin >> i_coord;
        if (i_coord == -1)
            goto reenter_pos;
        if (i_coord < 0 || i_coord >= 3)
            goto reenter_tile;
        std::cout << "your tile j coordinate (0-4): ";
        std::cin >> j_coord;
        if (j_coord < 0 || j_coord >= 5)
            goto reenter_tile;
        if (b.f[i_coord][j_coord] != 0) {
            std::cout << "position not empty\n";
            goto reenter_tile;
        }
        std::cout << "your tile (2 or 4): ";
        std::cin >> tile;
        if (tile != 2 && tile != 4)
            goto reenter_tile;
        bool possible = b.addTile(i_coord * 10 + j_coord * 2 + (tile == 2));
        assert(possible);
        std::cout << "position after tile addition: " << b << "\n";
        assert(get_user_value_from_disk(user_turn_positions, b));
        std::vector<int> ways;
        for (const std::string swipe : {"left", "right", "up", "down"}) {
            if (swipe == "left") {
                board temp(b);
                // std::cout << "possible to swipe left: " << temp.swipeLeft() << "\n";
                // std::cout << "if true, then the hater value of " << temp << " would be " << get_hater_value_from_disk(hater_turn_positions, temp) << "\n";
                // temp = b;
                if (temp.swipeLeft() && get_hater_value_from_disk(hater_turn_positions, temp))
                    ways.push_back(0);
            }
            if (swipe == "right") {
                board temp(b);
                // std::cout << "possible to swipe right: " << temp.swipeRight() << "\n";
                // std::cout << "if true, then the hater value of " << temp << " would be " << get_hater_value_from_disk(hater_turn_positions, temp) << "\n";
                // temp = b;
                if (temp.swipeRight() && get_hater_value_from_disk(hater_turn_positions, temp))
                    ways.push_back(1);
            }
            if (swipe == "up") {
                board temp(b);
                // std::cout << "possible to swipe up: " << temp.swipeUp() << "\n";
                // std::cout << "if true, then the hater value of " << temp << " would be " << get_hater_value_from_disk(hater_turn_positions, temp) << "\n";
                // temp = b;
                if (temp.swipeUp() && get_hater_value_from_disk(hater_turn_positions, temp))
                    ways.push_back(2);
            }
            if (swipe == "down") {
                board temp(b);
                // std::cout << "possible to swipe down: " << temp.swipeDown() << "\n";
                // std::cout << "if true, then the hater value of " << temp << " would be " << get_hater_value_from_disk(hater_turn_positions, temp) << "\n";
                // temp = b;
                if (temp.swipeDown() && get_hater_value_from_disk(hater_turn_positions, temp))
                    ways.push_back(3);
            }
        }
        std::cout << "ways: ";
        std::copy(ways.begin(), ways.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << "\n";
        assert(!ways.empty());
        int way = ways[rand() % ways.size()];
        switch (way) {
            case 0: assert(b.swipeLeft()); break;
            case 1: assert(b.swipeRight()); break;
            case 2: assert(b.swipeUp()); break;
            case 3: assert(b.swipeDown()); break;
        }
        assert(get_hater_value_from_disk(hater_turn_positions, b));
        std::cout << "position after answer: " << b << "\n";
    }
}
