// 1. 기본적인 내용들
// Host: CPU + RAM
// Device: GPU + VRAM
// host가 device한테 일 시킨다. 그런데 물리적으로 떨어져 있음. -> 데이터를 보내줘야 devcie가 일 할 수 있다

// step 1) Host -> Device
// step 2) kernel 가동
// step 3) Device -> Host

// RAM과 VRAM은 PCle가 연결하는데 느리다. -> CUDA 프로그래밍의 핵심은 RAM - VRAM 사이 이동 최소화.(한 번 보내고 뽕 뽑기)



// 2. devcie, host 변수 이름 규칙(convention) 및 함수 키워드
// h_A: host(CPU)의 A
// d_A: device(GPU)의 변수 A

// __global__
// host가 시키고 device가 일한다
// return: void

// __device__
// device가 시키고 device가 일한다
// return: 

// __host__
// host가 시키고 host가 일한다. (일반적인 C/C++ 코드)
// gpu랑 상관없음 표시하려고 씀. 꼭 안써도됨.



#include <stdio.h>
#include <cuda_runtime.h>

/*==== GPU에서 Block, Thread 위치 정보 출력하는 kernel ====*/

/* 궁극적으로 원하는 정보: Grid 안에서 내가 몇 번째 Thread 인가? */
// 큰 큐브 = grid, 중간 큐드 = Block, 작은 큐브 = Thread
// 큰 큐브는 중간 큐브들로, 중간 큐브는 작은 큐브들로 이루어져 있다
// 전략: 중간 큐브 단위로 크게 세고, 작은 큐브 단위로 작게 센 뒤, 단위통일.

// 큰 큐브 사이즈: (girdDim.x, gridDim.y, gridDim.z)
// 중간 큐브 사이즈: (blockDim.x, blockDim.y, blockDim,z)

// 큰 큐브 내 중간 큐브 위치: (blockIdx.x, blockIdx.y, blockIdx.z)
// 중간 큐브 내 작은 큐브 위치: (threadIdx.x, threadIdx.y, threadIdx.z)

__global__ void whoami(void) { // 원래 관습적으로 main함수에 __global__ 안붙인다
    // 1. Gird 안에서 Block 위치 (단위: Block 개수)
    // == 큰 큐브 안에서 내가 몇 번째 중간 큐브? (단위: 중간 규브)
    int block_id = 
        blockIdx.x + // 3. 길이 채운다
        blockIdx.y * gridDim.x + // 2. 넓이 채운다
        blockIdx.z * gridDim.y * gridDim.x; // 1. 부피 채운다

    // 2. Grid 안에서 내 Block 위치 (단위: Thread 개수)
    // == 큰 큐브 안에서 내가 몇 번째 중간 큐브? (단위: 작은 큐브)
    int block_offset = 
        block_id * // 지나온 block 의 개수
        blockDim.x * blockDim.y * blockDim.z; // block 당 thread 개수

    // 3. Block 안에서 내 Thread 위치 (단위: Thread 개수)
    // == 중간 큐브 안에서 내가 몇 번째 작은 큐브? (단위: 작은 큐브)
    int thread_offset = 
        threadIdx.x + // 3. 길이 채운다
        threadIdx.y * blockDim.x + // 2. 넓이 채운다
        threadIdx.z * blockDim.y * blockDim.x; // 1. 부피 채운다

    // 3. Grid 안에서 내 Thread 위치 (단위: Thread 개수)
    // == 큰 큐브 안에서 내가 몇 번째 작은 큐브? (단위: 작은 큐브)
    int id = block_offset + thread_offset;

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
    blockIdx.x, blockIdx.y, blockIdx.z, block_id,
    threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

__host__ int main(int argc, char **argv) {
    const int block_x = 2 , block_y = 8, block_z = 16; // block 256개
    // 한 block 당 thread 1024개 넣을 수 있다.(SM 하나에 CUDA 128개지만 1024개 thread가 와서 나머지 대기)
    const int thread_x = 8, thread_y = 8, thread_z = 8; // block 하나에 512 threads

    int num_blocks= block_x * block_y * block_z;
    int threads_per_block = thread_x * thread_y * thread_z;

    printf("%d blocks\n", num_blocks);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", num_blocks * threads_per_block);

    // dim 3: x, y ,z 값 담는 struct
    dim3 numBlocks(block_x, block_y, block_z);
    dim3 threadPerBlock(thread_x, thread_y, thread_z);

    // Kernel Launch
    // kernel 이름(함수명) <<< block 개수, block 당 thread 개수 >>> (arg1, arg2, ...);
    whoami <<< numBlocks, threadPerBlock >>> ();
    // 출력 131,072줄 찍힌다
    
    // cudaDeviceSynchronize(): GPU 끝날 때 까지 CPU가 기다려라.
    cudaDeviceSynchronize();
}