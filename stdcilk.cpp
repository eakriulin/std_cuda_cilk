#include <iostream>
#include <vector>
#include <cmath>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/cilkscale.h>
#include <chrono>
#include <cstdlib>

using namespace std;

const double EPSILON = 0.000001;

void calculateBucketResultForMean(vector<double>* dataset, vector<double>* results, int bucketIdx, int startIdx, int endIdx) {
    double bucketSum = 0.0;
    for (int i = startIdx; i < endIdx; ++i) {
        bucketSum += (*dataset)[i];
    }

    (*results)[bucketIdx] = bucketSum;
}

double calculateMeanConcurrently(vector<double>* dataset, int numberOfBuckets, int bucketSize) {
    vector<double> results(numberOfBuckets, 0.0);

    cilk_for (int bucketIdx = 0; bucketIdx < numberOfBuckets; ++bucketIdx) {
        int startIdx = bucketIdx * bucketSize;
        int endIdx = min(startIdx + bucketSize, static_cast<int>(dataset->size()));
        calculateBucketResultForMean(dataset, &results, bucketIdx, startIdx, endIdx);
    }

    double sum = 0.0;
    for (double result : results) {
        sum += result;
    }

    return sum / dataset->size();
}

void calculateBucketResultForSsd(vector<double>* dataset, vector<double>* results, double mean, int bucketIdx, int startIdx, int endIdx) {
    double bucketSsd = 0.0;
    for (int i = startIdx; i < endIdx; ++i) {
        bucketSsd += pow((*dataset)[i] - mean, 2);
    }

    (*results)[bucketIdx] = bucketSsd;
}

double calculateSsdConcurrently(vector<double>* dataset, double mean, int numberOfBuckets, int bucketSize) {
    vector<double> results(numberOfBuckets, 0.0);

    cilk_for (int bucketIdx = 0; bucketIdx < numberOfBuckets; ++bucketIdx) {
        int startIdx = bucketIdx * bucketSize;
        int endIdx = min(startIdx + bucketSize, static_cast<int>(dataset->size()));
        calculateBucketResultForSsd(dataset, &results, mean, bucketIdx, startIdx, endIdx);
    }

    double ssd = 0.0;
    for (double result : results) {
        ssd += result;
    }

    return ssd;
}

double calculateStdConcurrently(vector<double>* dataset) {
    int datasetSize = dataset->size();
    int numberOfThreads = __cilkrts_get_nworkers();
    int bucketSize = (datasetSize + numberOfThreads - 1) / numberOfThreads;
    int numberOfBuckets = (datasetSize + bucketSize - 1) / bucketSize;

    double mean = calculateMeanConcurrently(dataset, numberOfBuckets, bucketSize);
    double ssd = calculateSsdConcurrently(dataset, mean, numberOfBuckets, bucketSize);
    return sqrt(ssd / datasetSize);
}

double calculateStdSequentially(vector<double>* dataset) {
    int datasetSize = dataset->size();

    double sum = 0.0;
    for (int i = 0; i < datasetSize; ++i) {
        sum += (*dataset)[i];
    }

    double mean = sum / datasetSize;

    double ssd = 0.0;
    for (int i = 0; i < datasetSize; ++i) {
        ssd += pow(((*dataset)[i] - mean), 2);
    }

    return sqrt(ssd / datasetSize);
}

vector<double> prepare(int datasetSize) {
    cout << "[INFO]: dataset size " << datasetSize << ", preparing...\n";

    vector<double> dataset(datasetSize);
    for (int i = 0; i < datasetSize; ++i) {
        dataset[i] = static_cast<double>(rand()) / RAND_MAX * 100;
    }

    return dataset;
}

double compute(vector<double>* dataset) {
    wsp_t start = wsp_getworkspan();
    double std = calculateStdConcurrently(dataset);
    wsp_t end = wsp_getworkspan();

    wsp_dump(wsp_sub(end, start), "std");

    return std;
}

void report(vector<double>* dataset, double std) {
    cout << "[INFO]: computation completed - starting verification...\n";

    double expectedStd = calculateStdSequentially(dataset);
    if (abs(expectedStd - std) > EPSILON) {
        cout << "FAILED, results do not match — expectedStd = " << expectedStd << ", std = " << std << "\n";
    } else {
        cout << "SUCCESS, std = " << std << "\n";
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <datasetSize>\n";
        return 1;
    }

    int datasetSize = atoi(argv[1]);
    vector<double> dataset = prepare(datasetSize);
    double std = compute(&dataset);
    report(&dataset, std);

    return 0;
}
