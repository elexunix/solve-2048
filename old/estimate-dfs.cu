#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

const int threads = 4096;
const unsigned long long plays_per_thread = 1;

std::string itos(int a, int len = 0) {
    std::stringstream ss;
    ss << a;
    std::string s;
    ss >> s;
    while (s.length() < len)
        s = '0' + s;
    return s;
}

__global__ void init_random(curandState_t *curand_states, unsigned long long seq = 0) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < threads)
        curand_init(id, seq, 0, curand_states + id);
}

struct board {
    int f[4][4];
public:
    __device__ board(curandState_t& state) {
        /*for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                f[i][j] = 0;
        addTile(state);*/
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                f[i][j] = 1 << (1 + curand(&state) % 7);  // 2..127
    }
    __device__ bool addTile(int way) {  // way in [0;32)
        int& cell = f[(way&0xc)>>2][way&3];
        if (cell != 0)
            return false;
        cell = way & 0x10 ? 2 : 4;
        return true;
    }
    __device__ bool swipeLeft() {
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
    __device__ bool swipeRight() {
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
    __device__ bool swipeUp() {
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
    __device__ bool swipeDown() {
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

    __device__ bool operator== (const board& another) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] != another.f[i][j])
                    return false;
        return true;
    }
    __device__ bool operator!= (const board& another) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] != another.f[i][j])
                    return true;
        return false;
    }

    __device__ bool won() const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] == 256)
                    return true;
        return false;
    }
    __device__ bool finished() const {
        if (won())
            return true;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] == 0)
                    return false;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j)
                if (f[i][j] == f[i][j+1])
                    return false;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                if (f[i][j] == f[i+1][j])
                    return false;
        return true;
    }
    __device__ bool lost() const {
        return finished() && !won();
    }

    void print(std::ostream& out = std::cout) const {
        out << "board:\n";
        for (int i = 0; i < 4; ++i) {
            out << '\t';
            for (int j = 0; j < 4; ++j)
                out << f[i][j] << '\t';
            out << '\n';
        }
    }
};
struct diagram {
    unsigned long long success = 0, trials = 0;
    __device__ void clear() {
        success = trials = 0;
    }
    __device__ void add(bool ok) {
        success += ok;
        trials++;
    }
    bool empty() const {
        return trials == 0;
    }
    void print(std::ostream& out = std::cout) const {
        out << "diagram:\n";
        out << "\tsuccess:\t" << success << '\n';
        out << "\ttotal:\t" << trials << '\n';
        out.flush();
    }
    __host__ __device__ const diagram operator+ (const diagram& another) const {
        diagram result(*this);
        result.success += another.success;
        result.trials += another.trials;
        return result;
    }
    __host__ __device__ diagram& operator+= (const diagram& another) {
        return *this = *this + another;
    }
};

__device__ bool dfsProvesWin(const board& b, int depth) {
    if (depth == 0)
        return b.won();
    for (int swipe = 0; swipe < 4; ++swipe) {
        board temp(b);
        switch (swipe) {
            case 0: temp.swipeLeft(); break;
            case 1: temp.swipeRight(); break;
            case 2: temp.swipeUp(); break;
            case 3: temp.swipeDown(); break;
        }
        bool perfect_move = true;
        for (int ans = 0; ans < 32; ++ans) {
            board after_ans(temp);
            if (after_ans.addTile(ans))
                perfect_move &= dfsProvesWin(after_ans, depth - 1);
        }
        if (perfect_move)
            return true;
    }
    return false;
}
__device__ bool dfsProvesLoss(const board& b, int depth) {
    if (depth == 0)
        return b.lost();
    for (int swipe = 0; swipe < 4; ++swipe) {
        board temp(b);
        switch (swipe) {
            case 0: temp.swipeLeft(); break;
            case 1: temp.swipeRight(); break;
            case 2: temp.swipeUp(); break;
            case 3: temp.swipeDown(); break;
        }
        bool that_move_saves = true;
        for (int ans = 0; ans < 32; ++ans) {
            board after_ans(temp);
            if (after_ans.addTile(ans))
                that_move_saves &= !dfsProvesLoss(after_ans, depth - 1);
        }
        if (that_move_saves)
            return false;
    }
    return true;
}

__device__ bool statsOfRandomPlay(curandState_t& state, int depth = 5) {
    board b(state);
    return dfsProvesLoss(b, depth) || dfsProvesWin(b, depth);
}

__global__ void function(diagram *results, curandState_t *curand_states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= threads)
        return;
    results[id].clear();
    for (int i = 0; i < plays_per_thread; ++i)
        results[id].add(statsOfRandomPlay(curand_states[id]));
}


int main() {
    curandState_t *curand_states;
    cudaMallocManaged(&curand_states, threads * sizeof(curandState_t));
    init_random<<<threads/256+1,256>>>(curand_states, time(0));
    cudaDeviceSynchronize();
    diagram *results, results_global;
    cudaMallocManaged(&results, threads * sizeof(diagram));
    std::cout << "inited" << std::endl;
    long long prev = 0;
    while (/*results_global.trials < 100000 * threads * plays_per_thread*/ true) {
        function<<<threads/256+1,256>>>(results, curand_states);
        cudaDeviceSynchronize();
        for (int i = 0; i < threads; ++i)
            results_global += results[i];
        results_global.print();
        // if (results_global.trials > (prev << 30) + (1ll << 30)) {
        //     std::ofstream fout(("stats" + itos(++prev, 4) + ".txt").c_str());
        //     std::cout << "open: " << fout.is_open() << '\n';
        //     results_global.print_score_distribution(fout);
        // }
    }
    cudaFree(curand_states);
    cudaFree(results);
}

