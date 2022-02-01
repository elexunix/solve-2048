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


void get_tile(board &b) {
    while (true) {
        int i_coord, j_coord, tile;
        std::cout << "your tile i coordinate (0-3): ";
        std::cin >> i_coord;
        if (i_coord == -1)
            continue;  // reenter tile
        if (i_coord < 0 || i_coord >= 4)
            continue;  // reenter tile
        std::cout << "your tile j coordinate (0-3): ";
        std::cin >> j_coord;
        if (j_coord < 0 || j_coord >= 4)
            continue;  // reenter tile
        if (b.f[i_coord][j_coord] != 0) {
            std::cout << "position not empty\n";
            continue;  // reenter tile
        }
        std::cout << "your tile (2 or 4): ";
        std::cin >> tile;
        if (tile != 2 && tile != 4)
            continue;  // reenter tile
        bool possible = b.addTile(i_coord * 8 + j_coord * 2 + (tile == 2));
        assert(possible);
        std::cout << "position after tile addition: " << b << std::endl;
        return;
    }
}

void get_swipe(board &b) {
    while (true) {
        std::string swipe;
        std::cout << "your swipe (0-3, or left/right/up/down): ";
        std::cin >> swipe;
        board temp(b);
        if (swipe == "left" || swipe == "0") {
            if (temp.swipeLeft()) {
                b = temp;
                std::cout << "position after swipe: " << b << std::endl;
                return;
            }
            std::cout << "impossible to swipe left\n";
        } else if (swipe == "right" || swipe == "1") {
            if (temp.swipeRight()) {
                b = temp;
                std::cout << "position after swipe: " << b << std::endl;
                return;
            }
            std::cout << "impossible to swipe right\n";
        } else if (swipe == "up" || swipe == "2") {
            if (temp.swipeUp()) {
                b = temp;
                std::cout << "position after swipe: " << b << std::endl;
                return;
            }
            std::cout << "impossible to swipe up\n";
        } else if (swipe == "down" || swipe == "3") {
            if (temp.swipeDown()) {
                b = temp;
                std::cout << "position after swipe: " << b << std::endl;
                return;
            }
            std::cout << "impossible to swipe down\n";
        } else {
            std::cout << "not recognized\n";
        }
    }
}


void demonstrate_user_position_winnability_for_user(board);
void demonstrate_user_position_winnability_for_hater(board);
void demonstrate_hater_position_winnability_for_user(board);
void demonstrate_hater_position_winnability_for_hater(board);

void demonstrate_user_position_winnability_for_user(board b) {
    while (true) {
        assert(get_user_value_from_disk(b) == true);  // still winnable for user
        std::vector<int> ways;
        for (const std::string swipe : {"left", "right", "up", "down"}) {
            if (swipe == "left") {
                board temp(b);
                if (temp.swipeLeft() && get_hater_value_from_disk(temp))
                    ways.push_back(0);
            }
            if (swipe == "right") {
                board temp(b);
                if (temp.swipeRight() && get_hater_value_from_disk(temp))
                    ways.push_back(1);
            }
            if (swipe == "up") {
                board temp(b);
                if (temp.swipeUp() && get_hater_value_from_disk(temp))
                    ways.push_back(2);
            }
            if (swipe == "down") {
                board temp(b);
                if (temp.swipeDown() && get_hater_value_from_disk(temp))
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
        assert(get_hater_value_from_disk(b));
        std::cout << "position after swipe: " << b << std::endl;
        get_tile(b);
    }
}

void demonstrate_user_position_winnability_for_hater(board b) {
    assert(get_user_value_from_disk(b) == false);
    get_swipe(b);
    demonstrate_hater_position_winnability_for_hater(b);
}

void demonstrate_hater_position_winnability_for_user(board b) {
    assert(get_hater_value_from_disk(b) == true);
    get_tile(b);
    demonstrate_user_position_winnability_for_user(b);
}

void demonstrate_hater_position_winnability_for_hater(board b) {
    while (true) {
        assert(get_hater_value_from_disk(b) == false);
        std::vector<int> ways;
        for (int shot = 0; shot < 32; ++shot) {
            board temp(b);
            if (temp.addTile(shot) && !get_user_value_from_disk(temp))
                ways.push_back(shot);
        }
        std::cout << "ways: ";
        std::copy(ways.begin(), ways.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << "\n";
        assert(!ways.empty());
        int way = ways[rand() % ways.size()];
        assert(b.addTile(way));
        std::cout << "position after tile addition: " << b << std::endl;
        assert(!get_user_value_from_disk(b));
        get_swipe(b);
    }
}


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
    bool as_upos = get_user_value_from_disk(b), as_hpos = get_hater_value_from_disk(b);
    std::cout << "this user position is winnable for user: " << as_upos << "\n";
    std::cout << "this hater position is winnable for user: " << as_hpos << "\n";
    std::cout << "demonstrate first/second (1/2/q): ";
    std::string option;
    std::cin >> option;
    if (option == "quit" || option == "q") {
        goto reenter_pos;
    } else if (option == "first" || option == "1") {
        if (as_upos == true) {  // demonstrate how this user position is winnable, how to swipe
            demonstrate_user_position_winnability_for_user(b);
        } else {  // demonstrate how this user position is losable, how to put tiles evilly
            demonstrate_user_position_winnability_for_hater(b);
        }
    } else if (option == "second" || option == "2") {
        if (as_hpos == true) {  // demonstrate how this hater position is winnable, how to swipe
            demonstrate_hater_position_winnability_for_user(b);
        } else {  // demonstrate how this hater position is losable, how to put tiles evilly
            demonstrate_hater_position_winnability_for_hater(b);
        }
    }
    goto reenter_pos;
}
