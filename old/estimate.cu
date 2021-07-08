#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

const int threads = 4096;
const unsigned long long plays_per_thread = 1000;

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
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                f[i][j] = 0;
        addTile(state);
    }
    __device__ void addTile(curandState_t& state) {
        while (true) {
            unsigned int random_number = curand(&state);
            int& cell = f[(random_number&0xc)>>2][random_number&3];
            if (cell != 0)
                continue;
            cell = curand(&state) % 10 ? 2 : 4;
            return;
        }
    }
    __device__ int swipeLeft(curandState_t& state) {
        board prev(*this);
        int score = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0, w = 0; j < 4; ++j) {
                int tile = f[i][j];
                if (tile == 0)
                    continue;
                f[i][j] = 0;
                if (f[i][w] == 0)
                    f[i][w] = tile;
                else if (f[i][w] == tile)
                    score += f[i][w++] *= 2;
                else
                    f[i][++w] = tile;
            }
        }
        bool changed = *this != prev;
        if (changed)
            addTile(state);
        return score;
    }
    __device__ int swipeRight(curandState_t& state) {
        board prev(*this);
        int score = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 3, w = 3; j >= 0; --j) {
                int tile = f[i][j];
                if (tile == 0)
                    continue;
                f[i][j] = 0;
                if (f[i][w] == 0)
                    f[i][w] = tile;
                else if (f[i][w] == tile)
                    score += f[i][w--] *= 2;
                else
                    f[i][--w] = tile;
            }
        }
        bool changed = *this != prev;
        if (changed)
            addTile(state);
        return score;
    }
    __device__ int swipeUp(curandState_t& state) {
        board prev(*this);
        int score = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0, w = 0; j < 4; ++j) {
                int tile = f[j][i];
                if (tile == 0)
                    continue;
                f[j][i] = 0;
                if (f[w][i] == 0)
                    f[w][i] = tile;
                else if (f[w][i] == tile)
                    score += f[w++][i] *= 2;
                else
                    f[++w][i] = tile;
            }
        }
        bool changed = *this != prev;
        if (changed)
            addTile(state);
        return score;
    }
    __device__ int swipeDown(curandState_t& state) {
        board prev(*this);
        int score = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 3, w = 3; j >= 0; --j) {
                int tile = f[j][i];
                if (tile == 0)
                    continue;
                f[j][i] = 0;
                if (f[w][i] == 0)
                    f[w][i] = tile;
                else if (f[w][i] == tile)
                    score += f[w--][i] *= 2;
                else
                    f[--w][i] = tile;
            }
        }
        bool changed = *this != prev;
        if (changed)
            addTile(state);
        return score;
    }
    __device__ int swipe(curandState_t& state) {
        int mode = curand(&state) % 4;
        switch (mode) {
            case 0: return swipeLeft(state);
            case 1: return swipeRight(state);
            case 2: return swipeUp(state);
            case 3: return swipeDown(state);
        }
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
    __device__ bool finished() {
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
    struct stats {
        int power, score;
    };

    static const int p = 20;
    static const int S = 5000;
    unsigned long long by_powers[p], by_score[S], ttl = 0;
    diagram() {
        for (int i = 0; i < p; ++i)
            by_powers[i] = 0;
        for (int i = 0; i < S; ++i)
            by_score[i] = 0;
    }
    __device__ void clear() {
        for (int i = 0; i < p; ++i)
            by_powers[i] = 0;
        ttl = 0;
    }
    __device__ void add(int power, int score) {
        for (int i = 0; i < p; ++i)
            if (power == (1 << i))
                by_powers[i]++;
        if (score / 4 < S)
            by_score[score/4]++;
        ++ttl;
    }
    __device__ void add(stats s) {
        add(s.power, s.score);
    }
    bool empty() const {
        return ttl == 0;
    }
    void print(std::ostream& out = std::cout) const {
        out << "diagram:\n";
        if (empty()) {
            out << "\t[empty]\n";
            return;
        }
        for (int i = 0; i < p; ++i)
            if (by_powers[i] != 0)
                out << '\t' << (1 << i) << " -- " << by_powers[i] << '\n';
        out << "\t- total: " << ttl << '\n';
        out.flush();
    }
    void print_score_distribution(std::ostream& out = std::cout) const {
        for (int i = 0; i < S; ++i)
            out << i * 4 << '\t' << by_score[i] << '\n';
    }
    __host__ __device__ const diagram operator+ (const diagram& another) const {
        diagram result(*this);
        for (int i = 0; i < p; ++i)
            result.by_powers[i] += another.by_powers[i];
        for (int i = 0; i < S; ++i)
            result.by_score[i] += another.by_score[i];
        result.ttl += another.ttl;
        return result;
    }
    __host__ __device__ diagram& operator+= (const diagram& another) {
        return *this = *this + another;
    }
};

__device__ diagram::stats statsOfRandomPlay(curandState_t& state) {
    board b(state);
    int score = 0;
    do {
        score += b.swipe(state);
    } while (!b.finished());
    int top = -1;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            top = top > b.f[i][j] ? top : b.f[i][j];
    return {top, score};
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
    while (results_global.ttl < 100000 * threads * plays_per_thread) {
        function<<<threads/256+1,256>>>(results, curand_states);
        cudaDeviceSynchronize();
        for (int i = 0; i < threads; ++i)
            results_global += results[i];
        results_global.print();
        if (results_global.ttl > (prev << 30) + (1ll << 30)) {
            std::ofstream fout(("stats" + itos(++prev, 4) + ".txt").c_str());
            std::cout << "open: " << fout.is_open() << '\n';
            results_global.print_score_distribution(fout);
        }
    }
    cudaFree(curand_states);
    cudaFree(results);
}

