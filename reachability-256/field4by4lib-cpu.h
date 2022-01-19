#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>


std::string itos(long long a, int len = 0) {
    std::stringstream ss;
    ss << a;
    std::string s;
    ss >> s;
    while (s.length() < len)
        s = '0' + s;
    return s;
}
uint32_t hash(uint32_t x) {
  return std::hash<std::string>{}(std::to_string(x));  // std::hash<int> turned out to be identity...
}


const int win_const = 256;
const int log_win_const = 8;

struct board {
    int f[4][4];

public:
    board() {}

    int sum() const {
        int sum = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                sum += f[i][j];
        return sum;
    }

    bool swipeLeft() {
        board prev(*this);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0, w = 0; j < 4; ++j) {
                int tile = f[i][j];
                if (tile == 0)
                    continue;
                f[i][j] = 0;
                if (f[i][w] == 0)
                    f[i][w] = tile;
                else if (f[i][w] == tile)
                    f[i][w++] *= 2;
                else
                    f[i][++w] = tile;
            }
        }
        return *this != prev;
    }
    bool swipeRight() {
        board prev(*this);
        for (int i = 0; i < 4; ++i) {
            for (int j = 3, w = 3; j >= 0; --j) {
                int tile = f[i][j];
                if (tile == 0)
                    continue;
                f[i][j] = 0;
                if (f[i][w] == 0)
                    f[i][w] = tile;
                else if (f[i][w] == tile)
                    f[i][w--] *= 2;
                else
                    f[i][--w] = tile;
            }
        }
        return *this != prev;
    }
    bool swipeUp() {
        board prev(*this);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0, w = 0; j < 4; ++j) {
                int tile = f[j][i];
                if (tile == 0)
                    continue;
                f[j][i] = 0;
                if (f[w][i] == 0)
                    f[w][i] = tile;
                else if (f[w][i] == tile)
                    f[w++][i] *= 2;
                else
                    f[++w][i] = tile;
            }
        }
        return *this != prev;
    }
    bool swipeDown() {
        board prev(*this);
        for (int i = 0; i < 4; ++i) {
            for (int j = 3, w = 3; j >= 0; --j) {
                int tile = f[j][i];
                if (tile == 0)
                    continue;
                f[j][i] = 0;
                if (f[w][i] == 0)
                    f[w][i] = tile;
                else if (f[w][i] == tile)
                    f[w--][i] *= 2;
                else
                    f[--w][i] = tile;
            }
        }
        return *this != prev;
    }
    bool addTile(int way) {  // way in [0;32)
        int i = way / 8, j = way % 8 / 2;
        if (f[i][j] != 0)
            return false;
        f[i][j] = way & 1 ? 2 : 4;
        return true;
    }

    bool operator== (const board &another) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] != another.f[i][j])
                    return false;
        return true;
    }
    bool operator!= (const board &another) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] != another.f[i][j])
                    return true;
        return false;
    }

    bool won() const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] == win_const)
                    return true;
        return false;
    }

    friend std::ostream& operator<< (std::ostream &out, const board &b) {
        out << "[board:\n";
        for (int i = 0; i < 4; ++i) {
            out << "\t";
            std::copy(b.f[i], b.f[i] + 4, std::ostream_iterator<int>(std::cout, " "));
            std::cout << (i < 3 ? "\n" : "]");
        }
        return out;
    }
};


#include <ctime>

double start_time;
void report_start(const std::string what) {
    start_time = clock();
    std::cout << what << "...";
    std::cout.flush();
}
void report_finish() {
    std::cout << "\tdone in " << (clock() - start_time) / double(CLOCKS_PER_SEC) << std::endl;
}


int encode_cell(int value) {
    switch (value) {
        case 0: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        case 16: return 4;
        case 32: return 5;
        case 64: return 6;
        case 128: return 7;
        case 256: return 8;
    }
}
int decode_cell(int code) {
    switch (code) {
        case 0: return 0;
        case 1: return 2;
        case 2: return 4;
        case 3: return 8;
        case 4: return 16;
        case 5: return 32;
        case 6: return 64;
        case 7: return 128;
        case 8: return 256;
    }
}


// try to rewrite set_value, check performance
// try to rewrite operator[]s, check performance


bool get_value(const unsigned char *arr, long long pos) {
    return arr[pos >> 3] & 1 << (pos & 7);
}
void set_value(unsigned char *arr, long long pos, bool value) {
    if (value != get_value(arr, pos))
        arr[pos >> 3] ^= 1 << (pos & 7);
}


const int layers_cnt = 4 * 4 * win_const / 2 + 1;
long long index_dp[17][layers_cnt];

long long encode_inside_layer(const board &b, int sum) {
    // assert(b.sum() == sum);
    long long result = 0;
    int left = 15;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j, --left) {
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
    int left = 15;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j, --left) {
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


void write_arr_to_disk(long long n, const unsigned char *arr, const std::string filename) {
    // report_start("writing to \"" + filename + "\"");
    std::ofstream fout(filename, std::ios::out | std::ios::binary);
    fout.write(reinterpret_cast<const char*>(arr), n);
    // report_finish();
}
void read_arr_from_disk(long long n, unsigned char *arr, const std::string filename) {
    // report_start("reading \"" + filename + "\"");
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "UNABLE TO OPEN " << filename << "\n";
        // exit(1);
    }
    fin.read(reinterpret_cast<char*>(arr), n);
    // report_finish();
}

bool get_user_value_from_disk_slow(unsigned char *arr[layers_cnt], const board &b) /*[[deprecated]]*/ {
    arr[b.sum()] = new unsigned char[index_dp[16][b.sum()] / 8 + 1];
    read_arr_from_disk(index_dp[16][b.sum()] / 8 + 1, arr[b.sum()], "arrays4by4-256/ulayer" + itos(b.sum(), 4) + ".dat");
    bool result = get_value(arr, b);
    delete[] arr[b.sum()];
    return result;
}
bool get_user_value_from_disk_fast(const board &b) {
    std::ifstream in("arrays4by4-256/ulayer" + itos(b.sum(), 4) + ".dat");
    long long position_id = encode_inside_layer(b);
    in.seekg(position_id / 8);
    unsigned char ch;
    in >> ch;
    return ch & 1 << (position_id & 7);
}
bool get_user_value_from_disk(const board &b) {
    return get_user_value_from_disk_fast(b);
}
/*bool get_hater_value_from_disk_slow(unsigned char *arr[layers_cnt], const board &b) [[deprecated]] {  // impossible to provide unless recorded
    arr[b.sum()] = new unsigned char[index_dp[16][b.sum()] / 8 + 1];
    read_arr_from_disk(index_dp[16][b.sum()] / 8 + 1, arr[b.sum()], "arrays4by4-256/hlayer" + itos(b.sum(), 4) + ".dat");
    bool result = get_value(arr, b);
    delete[] arr[b.sum()];
    return result;
}*/
bool get_hater_value_from_disk(const board &b) {
    for (int way = 0; way < 32; ++way) {
        board temp(b);
        if (temp.addTile(way) && !get_user_value_from_disk(temp))
            return false;
    }
    return true;
}


void system(const std::string command) {
    system(command.c_str());
}


void fill_index_dp() {
    report_start("calculating index dp");
    // index_dp[cnt][sum] gives the number of ways to get sum in cnt cells (empty allowed)
    // useful for fast indexation inside layers of constant sum; checked to be at most ...
    index_dp[0][0] = 1;
    std::fill(index_dp[0] + 1, index_dp[0] + layers_cnt, 0);
    for (int cnt = 1; cnt <= 16; ++cnt) {
        for (int sum = 0; sum <= 4 * 4 * win_const / 2; ++sum) {
            index_dp[cnt][sum] = index_dp[cnt - 1][sum];
            for (int tile = 2; tile < win_const && tile <= sum; tile *= 2)
                index_dp[cnt][sum] += index_dp[cnt - 1][sum - tile];
        }
    }
    std::cout << "max size of layer: " << *std::max_element(index_dp[16], index_dp[16] + 4 * 4 * win_const / 2 + 1) << "\n";
    report_finish();
}

void report_final_result(unsigned char* user_turn_positions[layers_cnt], unsigned char* hater_turn_positions[layers_cnt]) {
    std::cout << "FINAL RESULT:\n";
    std::cout << "empty field:\n";
    board empty;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            empty.f[i][j] = 0;
    std::cout << "\tuser " << get_user_value_from_disk(empty) << ", ";
    std::cout << "hater " << get_hater_value_from_disk(empty) << "\n";
    std::cout << "field with just one tile:\n";
    for (int player = 0; player < 2; ++player) {
        std::cout << "\t" << (player == 0 ? "user" : "hater") << ":\n";
        for (int i = 0; i < 4; ++i) {
            std::cout << "\t";
            for (int j = 0; j < 4; ++j) {
                board b;
                for (int k = 0; k < 4; ++k)
                    for (int l = 0; l < 4; ++l)
                        b.f[k][l] = k == i && l == j ? 2 : 0;
                if (player == 0)
                    std::cout << get_user_value_from_disk(b);
                else
                    std::cout << get_hater_value_from_disk(b);
                b.f[i][j] = 4;
                if (player == 0)
                    std::cout << get_user_value_from_disk(b) << " ";
                else
                    std::cout << get_hater_value_from_disk(b) << " ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "field with two tiles:\n";
    for (int player = 0; player < 2; ++player) {
        std::cout << "\n\t" << (player == 0 ? "user" : "hater") << ":\n\n";
        for (int i1 = 0; i1 < 4; ++i1) {
            for (int i2 = 0; i2 < 4; ++i2) {
                std::cout << "\t";
                for (int j1 = 0; j1 < 4; ++j1) {
                    for (int j2 = 0; j2 < 4; ++j2) {
                        if (i1 == i2 && j1 == j2) {
                            std::cout << "---- ";
                            continue;
                        }
                        board b;
                        for (int k = 0; k < 4; ++k)
                            for (int l = 0; l < 4; ++l)
                                b.f[k][l] = 0;
                        for (int u1 = 2; u1 <= 4; u1 += 2) {
                            for (int u2 = 2; u2 <= 4; u2 += 2) {
                                b.f[i1][j1] = u1, b.f[i2][j2] = u2;
                                if (player == 0)
                                    std::cout << get_user_value_from_disk(b);
                                else
                                    std::cout << get_hater_value_from_disk(b);
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
}
