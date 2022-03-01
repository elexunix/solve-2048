#include <iostream>


const int win_const = 256;
const int log_win_const = 8;


template<int>
struct field_drawer;


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

    //friend std::ostream& operator<< (std::ostream &out, const board &b) {
    //    field_drawer<0>{}.draw(out, b);
    //}
    friend std::ostream& operator<< (std::ostream&, const board&);
};


/*template<int>
struct field_drawer {
  void print(std::ostream &out, const board &b) {
    out << "[board:\n";
    for (int i = 0; i < 4; ++i) {
        out << "\t";
        std::copy(b.f[i], b.f[i] + 4, std::ostream_iterator<int>(std::cout, "\t"));
        std::cout << (i < 3 ? "\n" : "]");
    }
  }
};*/
