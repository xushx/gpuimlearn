#include <stdio.h>
#include <cuda.h>
#include <numeric>
#include <stdlib.h>
#include <algorithm>
#include <cublas.h>

#define REDUCE_BLOCK_SIZE 32

__global__ void matrixMulKernel(float *m1, float *m2, float *r, int m1w, int m2w, int rw, int rh)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < rh) && (col < rw)) {
		// dot product
		float accum = 0.0f;
		for (int c = 0; c < m1w; c++)
		{
			float v1 = m1[row * m1w + c];
			float v2 = m2[c * m2w + col];
			accum += (v1 *  v2);
		}
		r[row * rw + col] = accum;
	}
}

__global__ void sigmoidKernel(float *r, int m, int flag=0)
{
    if(flag == 0){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < m) {
            float val = r[index];
            r[index] = 1.0 / (1.0 + expf(-val));
        }
    }else{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < m) {
            float val = r[index];
            float res = 1.0 / (1.0 + expf(-val));
            if(res >=0.01){
                r[index] = 1.0;
            }else{
                r[index] = 0.0;
            }
        }
    }
}

__global__ void updateParamsErrorKernel(float *p, float *ys, float *th, float *xs, int m, float alpha)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float h = *p;
		float y = *ys;
		float x = xs[index];

		th[index] = th[index] - alpha * (h - y) * x;
	}
}

__global__ void crossEntropyKernel(float *p, float *ys, float *r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float pval = p[index];
		float ysval = ys[index];

		r[index] = log1pf(expf(-ysval * pval));
	}
}

__global__ void reduceKernel(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * REDUCE_BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * REDUCE_BLOCK_SIZE;
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;
    if (start + REDUCE_BLOCK_SIZE + t < len)
       partialSum[REDUCE_BLOCK_SIZE + t] = input[start + REDUCE_BLOCK_SIZE + t];
    else
       partialSum[REDUCE_BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = REDUCE_BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];
}


extern "C"
{

// #define SAFE_CALL(value)

//Fit
void fit(float *x, float *y, float *p, int row, int col, int maxIterations=1000, float alpha=0.05) {

    // put stuff into gpu
	float *gpu_X;
	float *gpu_y;
	float *gpu_params;
	float *gpu_prediction;
	float *gpu_predictions;
	float *gpu_error;
	float *gpu_err_cost;

	int numOutputElements = 0;
	numOutputElements = row / (REDUCE_BLOCK_SIZE<<1);
	if (row % (REDUCE_BLOCK_SIZE<<1)) {
		numOutputElements++;
	}
    float* error_accum = new float[numOutputElements];
    float* predictions = new float[row] ();

	cudaMalloc((void**)&gpu_X, sizeof(float) * row * col);
	cudaMalloc((void**)&gpu_y, sizeof(float) * row);
	cudaMalloc((void**)&gpu_prediction, sizeof(float));
	cudaMalloc((void**)&gpu_predictions, sizeof(float) * row);
	cudaMalloc((void**)&gpu_error, sizeof(float) * row);
	cudaMalloc((void**)&gpu_params, sizeof(float) * col);
	cudaMalloc((void**)&gpu_err_cost, sizeof(float) * numOutputElements);

	cudaMemcpy(gpu_X, x, sizeof(float) * row * col, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y, y, sizeof(float) * row, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_params, p, sizeof(float) * col, cudaMemcpyHostToDevice);

	// invoke kernel
	const int blockWidth = 16;
	const int blockHeight = blockWidth;
	int numBlocksW = col / blockWidth;
	int numBlocksH = row / blockHeight;
	if (col % blockWidth) numBlocksW++;
	if (row % blockHeight) numBlocksH++;

	dim3 dimGrid(numBlocksW, numBlocksH);
	dim3 dimBlock(blockWidth, blockHeight);
	dim3 dimReduce((row - 1) / REDUCE_BLOCK_SIZE + 1);
	dim3 dimReduceBlock(REDUCE_BLOCK_SIZE);
	dim3 dimVectorGrid(((row - 1) / blockWidth * blockWidth) + 1);
	dim3 dimVectorBlock(blockWidth * blockWidth);

	for (int iter = 0; iter < maxIterations; ++iter) {
		for (int i = 0; i < row; ++i) {
			matrixMulKernel<<<dimGrid, dimBlock>>>(&gpu_X[i * col], gpu_params, gpu_prediction, col, 1, 1, 1);
			sigmoidKernel<<<dimVectorGrid, dimVectorBlock>>>(gpu_prediction, 1, 0);
			updateParamsErrorKernel<<<dimVectorGrid, dimVectorBlock>>>(gpu_prediction, &gpu_y[i], gpu_params, &gpu_X[i * col], col, alpha);
		}
		matrixMulKernel<<<dimGrid, dimBlock>>>(gpu_X, gpu_params, gpu_predictions, col, 1, 1, row);
		sigmoidKernel<<<dimVectorGrid, dimVectorBlock>>>(gpu_predictions, row, 0);
		// calculate error
		crossEntropyKernel<<<dimVectorGrid, dimVectorBlock>>>(gpu_predictions, gpu_y, gpu_error, row);
		reduceKernel<<<dimReduce, dimReduceBlock>>>(gpu_error, gpu_err_cost, row);
		// cudaMemcpy(error_accum, gpu_err_cost, sizeof(float) * numOutputElements, cudaMemcpyDeviceToHost);

	}

    cudaMemcpy(p, gpu_params, sizeof(float) * col, cudaMemcpyDeviceToHost);
    // cudaMemcpy(error_accum, gpu_err_cost, sizeof(float) * numOutputElements, cudaMemcpyDeviceToHost);

	delete[] error_accum;
	delete[] predictions;
	cudaFree(gpu_X);
	cudaFree(gpu_y);
	cudaFree(gpu_params);
	cudaFree(gpu_error);
	cudaFree(gpu_prediction);
	cudaFree(gpu_predictions);
	cudaFree(gpu_err_cost);
}

// Predicted
void predicted(float *te, float *labels, float *para, int trow, int tcol)
{
	float *gpu_te;
	float *gpu_l;
	float *gpu_p;

	cudaMalloc((void**)&gpu_te, sizeof(float) * trow * tcol);
	cudaMalloc((void**)&gpu_l, sizeof(float) * trow);
	cudaMalloc((void**)&gpu_p, sizeof(float) * tcol);

	cudaMemcpy(gpu_te, te, sizeof(float) * trow * tcol, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_l, labels, sizeof(float) * trow, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_p, para, sizeof(float) * tcol, cudaMemcpyHostToDevice);

	// invoke kernel
	const int blockWidth = 16;
	const int blockHeight = blockWidth;
	int numBlocksW = tcol / blockWidth;
	int numBlocksH = trow / blockHeight;
	if (tcol % blockWidth) numBlocksW++;
	if (trow % blockHeight) numBlocksH++;

	dim3 dimGrid(numBlocksW, numBlocksH);
	dim3 dimBlock(blockWidth, blockHeight);

	dim3 dimVectorGrid(((trow - 1) / blockWidth * blockWidth) + 1);
	dim3 dimVectorBlock(blockWidth * blockWidth);

    matrixMulKernel<<<dimGrid, dimBlock>>>(gpu_te, gpu_p, gpu_l, tcol, 1, 1, trow);
    sigmoidKernel<<<dimVectorGrid, dimVectorBlock>>>(gpu_l, trow, 1);

	// Copy DeviceToHost
	cudaMemcpy(labels, gpu_l, sizeof(float) * trow, cudaMemcpyDeviceToHost);

	cudaFree(gpu_te);
	cudaFree(gpu_l);
	cudaFree(gpu_p);
}

}
