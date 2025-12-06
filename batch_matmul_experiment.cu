#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define BATCH_SIZE 16
#define MATRIX_SIZE 4096
#define TILE_SIZE 32

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// NVML error checking macro
#define NVML_CHECK(call) \
    do { \
        nvmlReturn_t result = call; \
        if (result != NVML_SUCCESS) { \
            fprintf(stderr, "NVML error at %s:%d: %s\n", __FILE__, __LINE__, \
                    nvmlErrorString(result)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Structure for energy monitoring
typedef struct {
    nvmlDevice_t device;
    unsigned long long energy_start_mj;
    unsigned long long energy_end_mj;
} EnergyMonitor;

// Kernel (same as before)
__global__ void batchMatMulKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    int batch = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int batch_offset = batch * N * N;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        int aRow = row;
        if (aRow < N && aCol < N) {
            As[ty][tx] = A[batch_offset + aRow * N + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        int bCol = col;
        int bRow = t * TILE_SIZE + ty;
        if (bRow < N && bCol < N) {
            Bs[ty][tx] = B[batch_offset + bRow * N + bCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[batch_offset + row * N + col] = sum;
    }
}

void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

void startEnergyMeasurement(EnergyMonitor* monitor) {
    NVML_CHECK(nvmlDeviceGetTotalEnergyConsumption(monitor->device, &monitor->energy_start_mj));
}

void stopEnergyMeasurement(EnergyMonitor* monitor) {
    NVML_CHECK(nvmlDeviceGetTotalEnergyConsumption(monitor->device, &monitor->energy_end_mj));
}

nvmlDevice_t initNVML() {
    nvmlDevice_t device;
    NVML_CHECK(nvmlInit());
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device));

    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    NVML_CHECK(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
    printf("GPU Device: %s\n", name);

    unsigned long long test_energy;
    nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(device, &test_energy);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "ERROR: Hardware energy counter not supported on this GPU\n");
        fprintf(stderr, "This requires Volta (sm_70) or newer architecture\n");
        exit(EXIT_FAILURE);
    }
    printf("Hardware energy counter: Supported\n\n");

    return device;
}

// Calculate mean of an array
double calculateMean(double* values, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += values[i];
    }
    return sum / count;
}

// Calculate standard deviation
double calculateStdDev(double* values, int count, double mean) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = values[i] - mean;
        sum += diff * diff;
    }
    return sqrt(sum / count);
}

// Run experiment with specific number of iterations
void runExperiment(nvmlDevice_t nvmlDevice, float *d_A, float *d_B, float *d_C,
                   dim3 numBlocks, dim3 threadsPerBlock, int iterations, int num_trials) {

    double* energy_per_iter = (double*)malloc(num_trials * sizeof(double));
    double* total_energies = (double*)malloc(num_trials * sizeof(double));
    double* execution_times = (double*)malloc(num_trials * sizeof(double));

    printf("=== Experiment: %d iterations per trial, %d trials ===\n", iterations, num_trials);

    for (int trial = 0; trial < num_trials; trial++) {
        EnergyMonitor energyMonitor;
        energyMonitor.device = nvmlDevice;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Start measurement
        startEnergyMeasurement(&energyMonitor);
        CUDA_CHECK(cudaEventRecord(start));

        // Run kernel multiple times
        for (int i = 0; i < iterations; i++) {
            batchMatMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, MATRIX_SIZE);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Stop measurement
        stopEnergyMeasurement(&energyMonitor);

        // Calculate results
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        unsigned long long energy_consumed_mj = energyMonitor.energy_end_mj - energyMonitor.energy_start_mj;
        double total_energy_joules = energy_consumed_mj / 1000.0;
        double energy_per_iteration = total_energy_joules / iterations;

        energy_per_iter[trial] = energy_per_iteration;
        total_energies[trial] = total_energy_joules;
        execution_times[trial] = milliseconds;

        printf("  Trial %2d: Total=%.3f J, Per-iter=%.6f J, Time=%.2f ms\n",
               trial + 1, total_energy_joules, energy_per_iteration, milliseconds);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Small delay between trials
        usleep(100000); // 100ms
    }

    // Calculate statistics
    double mean_energy = calculateMean(energy_per_iter, num_trials);
    double stddev_energy = calculateStdDev(energy_per_iter, num_trials, mean_energy);
    double cv_energy = (stddev_energy / mean_energy) * 100.0; // Coefficient of Variation in %

    double mean_total = calculateMean(total_energies, num_trials);
    double stddev_total = calculateStdDev(total_energies, num_trials, mean_total);

    double mean_time = calculateMean(execution_times, num_trials);
    double stddev_time = calculateStdDev(execution_times, num_trials, mean_time);

    printf("\n--- Statistics ---\n");
    printf("Energy per iteration: %.6f ± %.6f J (CV: %.2f%%)\n",
           mean_energy, stddev_energy, cv_energy);
    printf("Total energy:         %.3f ± %.3f J\n", mean_total, stddev_total);
    printf("Execution time:       %.2f ± %.2f ms\n", mean_time, stddev_time);
    printf("Total duration:       %.2f ms (%.2f s)\n", mean_time, mean_time / 1000.0);

    // Find min and max
    double min_energy = energy_per_iter[0];
    double max_energy = energy_per_iter[0];
    for (int i = 1; i < num_trials; i++) {
        if (energy_per_iter[i] < min_energy) min_energy = energy_per_iter[i];
        if (energy_per_iter[i] > max_energy) max_energy = energy_per_iter[i];
    }
    printf("Range:                [%.6f, %.6f] J (%.2f%% variation)\n",
           min_energy, max_energy, ((max_energy - min_energy) / mean_energy) * 100.0);
    printf("\n");

    free(energy_per_iter);
    free(total_energies);
    free(execution_times);
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("GPU Energy Measurement Stability Experiment\n");
    printf("========================================\n\n");

    // Initialize NVML
    nvmlDevice_t nvmlDevice = initNVML();

    // Allocate and initialize data
    size_t matrixElements = MATRIX_SIZE * MATRIX_SIZE;
    size_t batchElements = BATCH_SIZE * matrixElements;
    size_t bytes = batchElements * sizeof(float);

    printf("Configuration:\n");
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Matrix size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("  Memory per batch: %.2f MB\n\n", bytes / (1024.0f * 1024.0f));

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    srand(time(NULL));
    initializeMatrix(h_A, batchElements);
    initializeMatrix(h_B, batchElements);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                   (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                   BATCH_SIZE);

    // Warm-up
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
        batchMatMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, MATRIX_SIZE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Warm-up complete.\n\n");

    // Experiment parameters
    int num_trials = 10;  // Number of trials per experiment
    int iteration_counts[] = {1, 2, 4, 8, 16};  // Different iteration counts to test
    int num_experiments = sizeof(iteration_counts) / sizeof(iteration_counts[0]);

    printf("Running experiments with %d trials each...\n\n", num_trials);

    // Run experiments
    for (int exp = 0; exp < num_experiments; exp++) {
        runExperiment(nvmlDevice, d_A, d_B, d_C, numBlocks, threadsPerBlock,
                     iteration_counts[exp], num_trials);

        // Longer delay between experiments
        if (exp < num_experiments - 1) {
            printf("Cooling down for 2 seconds...\n\n");
            sleep(2);
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    nvmlShutdown();

    printf("========================================\n");
    printf("Experiment complete!\n");
    printf("========================================\n");

    return 0;
}
