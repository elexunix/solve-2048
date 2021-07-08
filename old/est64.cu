#include <iostream>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

const int threads = 4096;
const unsigned long long plays_per_thread = 1024;

__global__ void init_random(curandState_t *curand_states, unsigned long long seq = 0) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < threads)
        curand_init(id, seq, 0, curand_states + id);
}

struct board {
    int f[3][3];
public:
    __device__ board(curandState_t& state) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                f[i][j] = 0;
        addTile(state);
    }
    __device__ void addTile(curandState_t& state) {
        while (true) {
            int& cell = f[curand(&state)%3][curand(&state)%3];
            if (cell != 0)
                continue;
            cell = curand(&state) % 10 ? 2 : 4;
            return;
        }
    }
    __device__ bool swipeLeft(curandState_t& state) {
        board prev(*this);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0, w = 0; j < 3; ++j) {
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
        bool changed = *this != prev;
        if (changed)
            addTile(state);
        return changed;
    }
    __device__ bool swipeRight(curandState_t& state) {
        board prev(*this);
        for (int i = 0; i < 3; ++i) {
            for (int j = 2, w = 2; j >= 0; --j) {
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
        bool changed = *this != prev;
        if (changed)
            addTile(state);
        return changed;
    }
    __device__ bool swipeUp(curandState_t& state) {
        board prev(*this);
        for (int i = 0; i < 3; ++i) {
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
        bool changed = *this != prev;
        if (changed)
            addTile(state);
        return changed;
    }
    __device__ bool swipeDown(curandState_t& state) {
        board prev(*this);
        for (int i = 0; i < 3; ++i) {
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
        bool changed = *this != prev;
        if (changed)
            addTile(state);
        return changed;
    }
    __device__ void swipe(curandState_t& state) {
        int mode = curand(&state) % 4;
        switch (mode) {
            case 0: swipeLeft(state); break;
            case 1: swipeRight(state); break;
            case 2: swipeUp(state); break;
            case 3: swipeDown(state); break;
        }
    }
    __device__ bool operator== (const board& another) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (f[i][j] != another.f[i][j])
                    return false;
        return true;
    }
    __device__ bool operator!= (const board& another) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (f[i][j] != another.f[i][j])
                    return true;
        return false;
    }
    __device__ bool finished() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (f[i][j] == 0)
                    return false;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 2; ++j)
                if (f[i][j] == f[i][j+1])
                    return false;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                if (f[i][j] == f[i+1][j])
                    return false;
        return true;
    }

    void print(std::ostream& out = std::cout) const {
        out << "board:\n";
        for (int i = 0; i < 3; ++i) {
            out << '\t';
            for (int j = 0; j < 3; ++j)
                out << f[i][j] << '\t';
            out << '\n';
        }
    }
};

__device__ int maxTileInRandomPlay(curandState_t& state) {
    board b(state);
    do {
        b.swipe(state);
    } while (!b.finished());
    int top = -1;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            top = top > b.f[i][j] ? top : b.f[i][j];
    return top;
}

struct diagram {
    static const int p = 20;
    unsigned long long by_powers[p], ttl = 0;
    diagram() {
        for (int i = 0; i < p; ++i)
            by_powers[i] = 0;
    }
    __device__ void clear() {
        for (int i = 0; i < p; ++i)
            by_powers[i] = 0;
        ttl = 0;
    }
    __device__ void add(int power) {
        for (int i = 0; i < p; ++i)
            if (power == (1 << i))
                by_powers[i]++;
        ++ttl;
    }
    bool empty() const {
        for (int i = 0; i < p; ++i)
            if (by_powers[i] != 0)
                return false;
        return true;
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
    __host__ __device__ const diagram operator+ (const diagram& another) const {
        diagram result(*this);
        for (int i = 0; i < p; ++i)
            result.by_powers[i] += another.by_powers[i];
        result.ttl += another.ttl;
        return result;
    }
    __host__ __device__ diagram& operator+= (const diagram& another) {
        return *this = *this + another;
    }
};

__global__ void function(diagram *results, curandState_t *curand_states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= threads)
        return;
    results[id].clear();
    for (int i = 0; i < plays_per_thread; ++i)
        results[id].add(maxTileInRandomPlay(curand_states[id]));
}


int main() {
    curandState_t *curand_states;
    cudaMallocManaged(&curand_states, threads * sizeof(curandState_t));
    init_random<<<threads/256+1,256>>>(curand_states, time(0));
    cudaDeviceSynchronize();
    diagram *results, results_global;
    cudaMallocManaged(&results, threads * sizeof(diagram));
    std::cout << "inited" << std::endl;
    while (results_global.ttl < 100000 * threads * plays_per_thread) {
        function<<<threads/256+1,256>>>(results, curand_states);
        cudaDeviceSynchronize();
        for (int i = 0; i < threads; ++i)
            results_global += results[i];
        results_global.print();
    }
    cudaFree(curand_states);
    cudaFree(results);
}

