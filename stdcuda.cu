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

// Prepare the dataset with random values
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

// Prepare the dataset with random values, calculateGlobalMean, calculateVariance as previously defined ...

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <dataSize>\n";
        return 1;
    }

    int dataSize = atoi(argv[1]);
    double* data = prepareUnified(dataSize);

    // Sequential calculation for comparison
    double seqMean, seqStdDev;
    calculateMeanAndStdDevSequentially(data, dataSize, seqMean, seqStdDev);
    cout << "Sequential Calculation:" << endl;
    cout << "Mean: " << seqMean << ", Standard Deviation: " << seqStdDev << endl;

    // Parallel CUDA calculation
    const int threadsPerBlock = 256; // Adjust based on GPU architecture and optimization
    double mean = calculateGlobalMean(data, dataSize, threadsPerBlock);
    double variance = calculateVariance(data, mean, dataSize, threadsPerBlock);
    double stdDev = sqrt(variance);

    cout << "Parallel CUDA Calculation:" << endl;
    cout << "Mean: " << mean << ", Standard Deviation: " << stdDev << endl;

    // Compare the results
    cout << "Comparison:" << endl;
    cout << "Difference in Mean: " << fabs(seqMean - mean) << endl;
    cout << "Difference in Standard Deviation: " << fabs(seqStdDev - stdDev) << endl;

    cudaFree(data); // Cleanup
    return 0;
}
