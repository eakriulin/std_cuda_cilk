#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

using namespace std;

// Kernel function to calculate sum of elements for mean calculation
__global__ void calculateGlobalMeanKernel(const double* data, double* partialSums, int dataSize) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < dataSize) ? data[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) partialSums[blockIdx.x] = sdata[0];
}

// Kernel function to calculate sum of squared differences for variance calculation
__global__ void calculateVarianceKernel(const double* data, double* partialVariances, double mean, int dataSize) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < dataSize) ? pow(data[i] - mean, 2) : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) partialVariances[blockIdx.x] = sdata[0];
}

// Function to calculate global mean
double calculateGlobalMean(const double* data, int dataSize, int threadsPerBlock) {
    int blocks = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
    double* partialSums;
    cudaMallocManaged(&partialSums, blocks * sizeof(double));

    calculateGlobalMeanKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(data, partialSums, dataSize);
    cudaDeviceSynchronize();

    double totalSum = 0;
    for (int i = 0; i < blocks; ++i) {
        totalSum += partialSums[i];
    }
    cudaFree(partialSums);

    return totalSum / dataSize;
}
// Function to calculate variance using the global mean
double calculateVariance(const double* data, double mean, int dataSize, int threadsPerBlock) {
    int blocks = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
    double* partialVariances;
    cudaMallocManaged(&partialVariances, blocks * sizeof(double));

    calculateVarianceKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(data, partialVariances, mean, dataSize);
    cudaDeviceSynchronize();

    double totalVariance = 0;
    for (int i = 0; i < blocks; ++i) {
        totalVariance += partialVariances[i];
    }
    cudaFree(partialVariances);

    return totalVariance / dataSize;
}


double* prepareUnified(int dataSize) {
    double* data;
    cudaMallocManaged(&data, dataSize * sizeof(double));
    for (int i = 0; i < dataSize; ++i) {
        data[i] = static_cast<double>(rand()) / RAND_MAX * 100.0;
    }
    return data;
}

// Function to calculate mean and standard deviation sequentially for correctness checking
void calculateMeanAndStdDevSequentially(const double* data, int dataSize, double &mean, double &stdDev) {
    double sum = 0.0;
    for (int i = 0; i < dataSize; ++i) {
        sum += data[i];
    }
    mean = sum / dataSize;

    double varianceSum = 0.0;
    for (int i = 0; i < dataSize; ++i) {
        varianceSum += (data[i] - mean) * (data[i] - mean);
    }
    double variance = varianceSum / dataSize;
    stdDev = sqrt(variance);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <dataSize> <threadsPerBlock>\n";
        return 1;
    }

    int dataSize = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);

    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
        cerr << "Error: threadsPerBlock must be between 1 and 1024.\n";
        return 1;
    }

    // Prepare the dataset
    double* data = prepareUnified(dataSize);

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Execute the kernel to calculate the global mean
    double mean = calculateGlobalMean(data, dataSize, threadsPerBlock);
    cudaDeviceSynchronize(); // Ensure mean calculation completion

    // Execute the kernel to calculate variance using the global mean
    double variance = calculateVariance(data, mean, dataSize, threadsPerBlock);
    cudaDeviceSynchronize(); // Ensure variance calculation completion

    // Calculate the standard deviation from the variance
    double stdDev = sqrt(variance);


    // Record the stop event and calculate the elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Parallel CUDA Calculation:" << endl;
    cout << "Mean: " << mean << ", Standard Deviation: " << stdDev << endl;
    cout << "Calculation took " << milliseconds << " milliseconds." << endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(data);

    return 0;
}
