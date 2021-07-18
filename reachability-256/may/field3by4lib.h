#include <fstream>
#include <iostream>
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
    int f[3][4];
public:
    __host__ __device__ board() {}

    __host__ __device__ int sum() const {
        int sum = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                sum += f[i][j];
        return sum;
    }

    __host__ __device__ bool swipeLeft() {
        board prev(*this);
        for (int i = 0; i < 3; ++i) {
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
    __host__ __device__ bool swipeRight() {
        board prev(*this);
        for (int i = 0; i < 3; ++i) {
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
    __host__ __device__ bool swipeUp() {
        board prev(*this);
        for (int i = 0; i < 4; ++i) {
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
    __host__ __device__ bool swipeDown() {
        board prev(*this);
        for (int i = 0; i < 4; ++i) {
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
    __host__ __device__ bool addTile(int way) {  // way in [0;24)
        int i = way >> 3, j = way >> 1 & 3;
        if (f[i][j] != 0)
            return false;
        f[i][j] = way & 1 ? 2 : 4;
        return true;
    }

    __host__ __device__ bool operator== (const board& another) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] != another.f[i][j])
                    return false;
        return true;
    }
    __host__ __device__ bool operator!= (const board& another) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] != another.f[i][j])
                    return true;
        return false;
    }

    __host__ __device__ bool won() const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] == win_const)
                    return true;
        return false;
    }
    /*__host__ __device__ bool finished() const {
        if (won())
            return true;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] == 0)
                    return false;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (f[i][j] == f[i][j + 1])
                    return false;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] == f[i + 1][j])
                    return false;
        return true;
    }
    __host__ __device__ bool lost() const {
        return finished() && !won();
    }*/

    void print(std::ostream& out = std::cout) const {
        out << "board:\n";
        for (int i = 0; i < 3; ++i) {
            out << '\t';
            for (int j = 0; j < 4; ++j)
                out << f[i][j] << '\t';
            out << '\n';
        }
    }
};


#include <ctime>

double start_time;
void report_start(std::string what) {
    start_time = clock();
    std::cout << what << "...";
    std::cout.flush();
}
void report_finish() {
    std::cout << "\tdone in " << (clock() - start_time) / double(CLOCKS_PER_SEC) << std::endl;
}


__host__ __device__ int encode_cell(int value) {
    switch (value) {
        case 0: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        case 16: return 4;
        case 32: return 5;
        case 64: return 6;
    }
}
__host__ __device__ int decode_cell(int code) {
    switch (code) {
        case 0: return 0;
        case 1: return 2;
        case 2: return 4;
        case 3: return 8;
        case 4: return 16;
        case 5: return 32;
        case 6: return 64;
    }
}
// __host__ __device__ long long encode_position(const board& b) {
//     long long result = 0;
//     for (int i = 0; i < 3; ++i)
//         for (int j = 0; j < 4; ++j)
//             result = result * log_win_const + encode_cell(b.f[i][j]);
//     return result;
// }
// __host__ __device__ const board decode_position(long long id) {
//     board result;
//     for (int i = 2; i >= 0; --i)
//         for (int j = 3; j >= 0; --j)
//             result.f[i][j] = decode_cell(id % log_win_const), id /= log_win_const;
//     return result;
// }


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}