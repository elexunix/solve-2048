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


const int win_const = 128;
const int log_win_const = 7;

struct board {
    int f[3][5];

public:
    board() {}

    int sum() const {
        int sum = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 5; ++j)
                sum += f[i][j];
        return sum;
    }

    bool swipeLeft() {
        board prev(*this);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0, w = 0; j < 5; ++j) {
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
        for (int i = 0; i < 3; ++i) {
            for (int j = 4, w = 4; j >= 0; --j) {
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
        for (int i = 0; i < 5; ++i) {
            for (int j = 0, w = 0; j < 3; ++j) {
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
        for (int i = 0; i < 5; ++i) {
            for (int j = 2, w = 2; j >= 0; --j) {
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
    bool addTile(int way) {  // way in [0;30)
        int i = way / 10, j = way % 10 / 2;
        if (f[i][j] != 0)
            return false;
        f[i][j] = way & 1 ? 2 : 4;
        return true;
    }

    bool operator== (const board &another) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 5; ++j)
                if (f[i][j] != another.f[i][j])
                    return false;
        return true;
    }
    bool operator!= (const board &another) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 5; ++j)
                if (f[i][j] != another.f[i][j])
                    return true;
        return false;
    }

    bool won() const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 5; ++j)
                if (f[i][j] == win_const)
                    return true;
        return false;
    }

    friend std::ostream& operator<< (std::ostream &out, const board &b) {
        out << "[board:\n";
        for (int i = 0; i < 3; ++i) {
            out << "\t";
            std::copy(b.f[i], b.f[i] + 5, std::ostream_iterator<int>(std::cout, " "));
            std::cout << (i < 2 ? "\n" : "]");
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
    }
}
