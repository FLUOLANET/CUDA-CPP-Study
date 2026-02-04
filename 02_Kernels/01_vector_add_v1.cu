#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/*==== CPU 함수 vs CUDA Kerenl 속도(실행 시간) 비교 ====*/

#define N 10000000 // vector 사이즈 1000만
#define BLOCK_SIZE 256 // block 사이즈는 256



// 1. CPU가 직접 vector 덧셈 하는 함수
__host__ void h_vector_add(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i]; 
        // *(c + i) = *(a + i) + *(b + i) 와 같음
    }
}

// 2. GPU에게 vector addition 맡기는 CUDA Kernel(함수)

// - SPMD(Single Program, Multiple Data)
//  : CUDA Kernel 하나를 보고 여러 core들이 각자의 blockIdx, threadIdx에 맞게 자신의 정보를 주입해 작업
__global__ void d_vector_add(float *a, float *b, float *c, int n) {

    // i는 core 자기 자신의 정보
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 자신의 i가 n보다 작으면 자기 i에 맞는 메모리 찾아가서 더하는 작업 해라
    if (i < n) {
        c[i] = a[i] + b[i];    
    }
}

// 3. 랜덤한 값들로 벡터 초기화하는 함수
void init_vector(float *init_vec, int n) {
    for (int i = 0; i < n; i++) {
        init_vec[i] = (float)rand() / RAND_MAX; // float in [0, 1]
    }
}

// 4. 실행 시간 측정하는 함수
double get_time() {
    struct timespec ts; 
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(int argc, char **argv) {
    // 인자로 넣을 벡터들
    float *h_a, *h_b, *h_c, *h_c_cpu, *h_c_gpu; // gpu 계산결과 cpu로 복사해야됨
    float *d_a, *d_b, *d_c;

    // host memory 할당
    size_t size = N * sizeof(float);
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    // 난수 설정 후 vector 초기화
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // device memory 할당
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // device memory에 host memory에 만들었던 vector 복사
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // gird, block 차원 정의
    // int num_blocks = (N / BLOCK_SIZE) + 1 
    // -> 이렇게 하면 N이 나누어 떨어졌을 때 필요없는 block 생김
    // N = p * BLOCK_SIZE + q 꼴일 때, 아래와 같이 하면 q >= 1 일 때만 block 하나 더 가능
    int n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
} 