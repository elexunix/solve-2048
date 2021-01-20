#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ __device__ long long factorial(int n) {
    switch (n) {
        case 0: return 1;
        case 1: return 1;
        case 2: return 2;
        case 3: return 6;
        case 4: return 24;
        case 5: return 120;
        case 6: return 720;
        case 7: return 5040;
        case 8: return 40320;
        case 9: return 362880;
        case 10: return 3628800;
        case 11: return 39916800;
        case 12: return 479001600;
        case 13: return 6227020800;
        case 14: return 87178291200;
        case 15: return 1307674368000;
        case 16: return 20922789888000;
    }
}

std::vector<int> temp;
struct arrangement {
    static const int N = 9;
    int cnt_of_tile[N], tiles_used = 0;
    long long cnt_positions;
    __host__ __device__ int operator[] (int index) const {
        return cnt_of_tile[index];
    }
    __host__ __device__ int& operator[] (int index) {
        return cnt_of_tile[index];
    }
    bool operator< (const arrangement& another) const {
        for (int i = 0; i < N; ++i)
            if (cnt_of_tile[i] != another[i])
                return cnt_of_tile[i] < another[i];
        return false;
    }
};
int arrangements_ptr = 0;
long long max_arrangement_size;

long long cnt_positions_with_sum(int field_size, int tiles, int rest_cnt, int rest_sum, std::vector<int>& cnt_by_tile,
        arrangement *arrangements) {  // tiles = 7 means 2, 4, ..., 128 available
    if (cnt_by_tile.size() == tiles) {
        if (rest_sum > 0)
            return 0;
        long long cnt = factorial(field_size) / factorial(rest_cnt);
        for (int x: cnt_by_tile)
            cnt /= factorial(x);
        for (int i = 0; i < tiles; ++i)
            arrangements[arrangements_ptr][i] = cnt_by_tile[i];
        arrangements[arrangements_ptr].tiles_used = field_size - rest_cnt;
        arrangements[arrangements_ptr++].cnt_positions = cnt;
        max_arrangement_size = std::max(max_arrangement_size, cnt);
        return cnt;
    }
    long long result = 0;
    for (int k = 0; k <= rest_cnt && k << cnt_by_tile.size() + 1 <= rest_sum; ++k) {
        std::cout.flush();
        cnt_by_tile.push_back(k);
        result += cnt_positions_with_sum(field_size, tiles, rest_cnt - k, rest_sum - (k << cnt_by_tile.size()), cnt_by_tile, arrangements);
        cnt_by_tile.pop_back();
    }
    return result;
}

__global__ void compute(arrangement *arrangements, int cnt, int field_size, long long *links, long long *links_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cnt)
        return;
    int tiles = arrangements[id].tiles_used;
    links[id] = links_size[id] = 0;
    if (tiles > 0) {
        int expand[100];
        for (int i = 0; i < tiles; ++i)
            expand[i] = 0;
        while (expand[0] < 2) {
            int expanded_cnt = tiles;
            for (int i = 0; i < tiles; ++i)
                expanded_cnt += expand[i];
            if (expanded_cnt <= field_size)
                ++links[id];
            expand[tiles - 1]++;
            for (int i = tiles - 1; i > 0; --i)
                if (expand[i] == 2)
                    expand[i] = 0, expand[i - 1]++;
        }
    }
}

int main() {
    arrangement *arrangements;
    const int threads = 800000;
    cudaMallocManaged(&arrangements, threads * sizeof(arrangement));
    int field_size, tiles;
    std::cout << "field size: ";
    std::cin >> field_size;
    std::cout << "tiles up to 2**";
    std::cin >> tiles;
    long long max_layer_size = -1, total_size = 0;
    for (int sum = 0; sum <= field_size << tiles; sum += 2) {
        std::cout << "layer " << sum << ", size ";
        long long layer_size = cnt_positions_with_sum(field_size, tiles, field_size, sum, temp, arrangements);
        max_layer_size = std::max(max_layer_size, layer_size);
        total_size += layer_size;
        std::cout << layer_size << '\n';
    }
    std::cout << "total size: " << total_size << '\n';
    std::cout << "max layer size: " << max_layer_size << '\n';
    std::cout << "arrangements: " << arrangements_ptr << '\n';
    std::cout << "max arrangement size: " << max_arrangement_size << std::endl;
    std::sort(arrangements, arrangements + arrangements_ptr);
    long long *links, *links_size;
    cudaMallocManaged(&links, threads * sizeof(long long));
    cudaMallocManaged(&links_size, threads * sizeof(long long));
    compute<<<800/256+1,256>>>(arrangements, arrangements_ptr, field_size, links, links_size);
    gpuErrchk(cudaDeviceSynchronize());
    long long total_links = std::accumulate(links, links + arrangements_ptr, 0), max_links = *std::max_element(links, links + arrangements_ptr);
    std::cout << "total links: " << total_links << '\n';
    std::cout << "max links per node: " << max_links << '\n';
    long long total_links_size = std::accumulate(links_size, links_size + arrangements_ptr, 0),
        max_links_size = *std::max_element(links_size, links_size + arrangements_ptr);
    std::cout << "total links size: " << total_links_size << '\n';
    std::cout << "max links size: " << max_links_size << '\n';
    cudaFree(arrangements);
    cudaFree(links);
    cudaFree(links_size);
}