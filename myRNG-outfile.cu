#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <bitset>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>
#include <chrono>
#include <random>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)
#define N 128
using namespace std;

__global__ void my_generateRandom(unsigned int *result)
{
	//thread index
	int id = threadIdx.x + blockIdx.x * N;
	/* Copy state to local memory for efficiency */
	unsigned y = result[id];

	//Xorshift
	y = y ^ (y << 11);
	y = y ^ (y >> 7);
	y = y ^ (y >> 12);

	/* Copy state back to global memory */
	result[id] = y;
}

int main(int argc, char *argv[])
{
	int i;
	unsigned int total;
	unsigned int *devResults, *hostResults;

	/* Allocate space for results on host */
	hostResults = (unsigned int *)calloc(N * N, sizeof(int));

	/* Allocate space for results on device */
	CUDA_CALL(cudaMalloc((void **)&devResults, N * N *
		sizeof(unsigned int)));

	/* Set results and seed to 0 */
	CUDA_CALL(cudaMemset(devResults, 0, N * N *
		sizeof(unsigned int)));

	/* Setup prng states */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	mt19937 rand_num(seed);
	
	unsigned x = 123456789, y = 362436069, z = 521288629,
		w = 88675123, v = 5783321, d = 6615241;
	unsigned t;
	for (int i = 0; i < N * N; i++) {
		t = (x^(x >> 2));
		x = y;
		y = z;
		z = w;
		w = v;
		v = (v^(v << 4))^(t^(t << 1)); 
		//hostResults[i] = (d += 362437) + v;
		hostResults[i] = rand_num();
		rand_num();
	}
	CUDA_CALL(cudaMemcpy(devResults, hostResults, N * N *
		sizeof(unsigned int), cudaMemcpyHostToDevice));

	/*open file*/
	ofstream outfile;
	outfile.open("rng-test.txt");

	/* Generate and use pseudo-random  */
	for (i = 0; i < 128; i++) {
		my_generateRandom << <N, N >> > (devResults);

		/* Copy device memory to host */
		CUDA_CALL(cudaMemcpy(hostResults, devResults, N * N *
			sizeof(unsigned int), cudaMemcpyDeviceToHost));
		for (int j = 0; j < N * N; j++) {
			bitset<32> t(hostResults[j]);
			outfile << t;
		}
	}

	outfile.close();

	/* Cleanup */
	CUDA_CALL(cudaFree(devResults));
	free(hostResults);
	printf("^^^^ kernel_example PASSED\n");
	return EXIT_SUCCESS;
}