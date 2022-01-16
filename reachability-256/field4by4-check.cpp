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
#include "field4by4lib-cpu.h"


int main() {
    fill_index_dp();

    unsigned char *user_turn_positions[layers_cnt], *hater_turn_positions[layers_cnt];  // sized index_dp[16][sum]/8+1
    // values are always _for user_; for example, a unit in hater's array means the position is winnable by user, same as with user's array

    report_final_result(user_turn_positions, hater_turn_positions);

    system("date");

    std::cout << "Initial query: enter some position id, this program will show this position (can be ignored)\n";
    int entered_layer;
    long long entered_id;
    std::cout << "\tsum (layer): ";
    std::cin >> entered_layer;
    std::cout << "\tposition id (up to " << index_dp[16][entered_layer] - index_dp[16][entered_layer - 1] << "): ";
    std::cin >> entered_id;
    std::cout << decode_inside_layer(entered_id, entered_layer) << std::endl;

    reenter_pos:

    std::cout << "Enter position:\n";
    board b;
    for (int i = 0; i < 4; ++i) {
        std::cout << '\t';
        for (int j = 0; j < 4; ++j)
            std::cin >> b.f[i][j];
    }
    std::cout << "this user position is winnable for user: " << get_user_value_from_disk(user_turn_positions, b) << "\n";
    std::cout << "this hater position is winnable for user: " << get_hater_value_from_disk(hater_turn_positions, b) << "\n";
    if (!get_hater_value_from_disk(hater_turn_positions, b))
        goto reenter_pos;

    while (true) {
        reenter_tile:
        int i_coord, j_coord, tile;
        std::cout << "your tile i coordinate (0-3): ";
        std::cin >> i_coord;
        if (i_coord == -1)
            goto reenter_pos;
        if (i_coord < 0 || i_coord >= 4)
            goto reenter_tile;
        std::cout << "your tile j coordinate (0-3): ";
        std::cin >> j_coord;
        if (j_coord < 0 || j_coord >= 4)
            goto reenter_tile;
        if (b.f[i_coord][j_coord] != 0) {
            std::cout << "position not empty\n";
            goto reenter_tile;
        }
        std::cout << "your tile (2 or 4): ";
        std::cin >> tile;
        if (tile != 2 && tile != 4)
            goto reenter_tile;
        bool possible = b.addTile(i_coord * 8 + j_coord * 2 + (tile == 2));
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
