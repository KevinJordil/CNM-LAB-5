/*
Description:
  The MLP architecture is a 3-layer neural network with 1 hidden layer.
  The input layer X has 784 nodes (28x28 images)
  The hidden layer H has 1000 nodes
  The output layer Y has 10 nodes (0-9 digits)
  The batch size is 8.
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "io.h"
#include "rand.h"
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* batch size */
#define B 10
/* input size */
#define X 784
/* hidden size */
#define H 1000
/* output size */
#define Y 10

#define NUM_THREADS 4
#define ITERATIONS 1000000
#define TARGET_ACC 0.95f
#define STATS_INTERVAL 50000
#define SMOOTHING 0.99999f
#define LEARNING_RATE 5 * 1e-4f
#define DATAPOINTS 50000
#define WEIGHT_DECAY .0f
#define DROPOUT 0.0
#define LOGISTIC 0
#define RELU 1
#define TANH 0

extern void randn(float *out, float mean, float std, int n);

unsigned char inputs[X * DATAPOINTS];
unsigned char labels[DATAPOINTS];

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
}

/* Cuda check error */
void cuda_check_error(cudaError err)
{
  if (err != cudaSuccess)
  {
    printf("CUDA error (%d): %s  \n", err, cudaGetErrorString(err));
    exit(-1);
  }
}

// Cuda kernel function of loop
__global__ void backprop_kernel(float *p, float *t, float *dv, float *v, float *dh, float *h)
{
  int y_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int h_idx = threadIdx.y + blockIdx.y * blockDim.y;

  if (y_idx >= H || h_idx >= Y)
    return;

  dv[h_idx * H + y_idx] = 0.0f;
  dh[h_idx * H + y_idx] = 0.0f;

  __syncthreads();

  float dy, dh_gpu, dv_gpu;

  for (int b_idx = 0; b_idx < B; b_idx++)
  {
    dy = p[b_idx * Y + h_idx] - t[b_idx * Y + h_idx];
    dv_gpu = h[b_idx * H + y_idx] * dy;
    dh_gpu = v[h_idx * H + y_idx] * dy;
    atomicAdd(&dv[h_idx * H + y_idx], dv_gpu);
    atomicAdd(&dh[b_idx * H + y_idx], dh_gpu);
  }
}

//     /* nonlinearity on h */
//     // #pragma omp parallel for
//     for (int j = 0; j < H * B; j++)
// #if LOGISTIC
//       dh[j] = dh[j] * h[j] * (1.0f - h[j]);
// #endif
// #if RELU
//     dh[j] = dh[j] * h[j];
// #endif
// #if TANH
//     dh[j] = dh[j] * (1.0f - h[j] * h[j]);
// #endif

__global__ void nonlinearity_kernel(float *dh, float *h)
{
  int h_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (h_idx >= H * B)
    return;

  dh[h_idx] = dh[h_idx] * h[h_idx];
}

__global__ void set_dw_zero_kernel(float *dw)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < H && idy < X)
  {
    dw[idx * X + idy] = 0.0f;
  }
}

// for (int j = 0; j < H; j++)
// {
//   for (int i = 0; i < X; i++)
//   {
//     for (int b = 0; b < B; b++)
//     {
//       dw[j * X + i] += x[b * X + i] * dh[b * H + j];
//     }
//   }
// }
__global__ void update_weights_kernel(float *dw, float *x, float *dh)
{
  int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int h_idx = threadIdx.y + blockIdx.y * blockDim.y;

  if (x_idx >= X || h_idx >= H)
    return;

  //dw[h_idx * X + x_idx] = 0.0f;

  //__syncthreads();

  float dw_gpu;

  for (int b_idx = 0; b_idx < B; b_idx++)
  {
    dw_gpu = x[b_idx * X + x_idx] * dh[b_idx * H + h_idx];
    atomicAdd(&dw[h_idx * X + x_idx], dw_gpu);
  }
}

int main(int argc, char **argv)
{
  /* command line argument */
  if (argc > 1)
    if (0 == strcmp(argv[1], "help"))
    {
      printf("usage: %s max_iters lr decay\n", argv[0]);
      return 0;
    }

  omp_set_num_threads(NUM_THREADS);

  /* x -w-> h -v-> y */
  float *x, *h, *y, *p, *t, *c; /*states*/
  float *w, *v;                 /*weights*/
  float *dh, *dy;               /*states-grads*/
  float *dw, *dv;               /*weight-grads*/
  float *m;                     /*dropout*/

  /* allocate memory for arrays */
  w = (float *)malloc(sizeof(float) * X * H);
  m = (float *)malloc(sizeof(float) * H * B);
  y = (float *)malloc(sizeof(float) * Y * B);
  c = (float *)malloc(sizeof(float) * Y * B);

  /* Initialize memory for GPU */
  cuda_check_error(cudaMallocManaged(&x, X * B * sizeof(float)));
  cuda_check_error(cudaMallocManaged(&dw, X * H * sizeof(float)));
  cuda_check_error(cudaMallocManaged(&dy, B * Y * sizeof(float)));
  cuda_check_error(cudaMallocManaged(&dv, Y * H * sizeof(float)));
  cuda_check_error(cudaMallocManaged(&dh, B * H * sizeof(float)));
  cuda_check_error(cudaMallocManaged(&p, B * Y * sizeof(float)));
  cuda_check_error(cudaMallocManaged(&t, B * Y * sizeof(float)));
  cuda_check_error(cudaMallocManaged(&h, B * H * sizeof(float)));
  cuda_check_error(cudaMallocManaged(&v, Y * H * sizeof(float)));

  dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(1, 1, 1);

  /* init stats */
  float smooth_act = 0.0f;
  float smooth_ce = logf(Y);
  float smooth_acc = 1.0f / Y;

  /* set default values for command line arguments */
  int max_iters = argc > 1 ? atoi(argv[1]) : ITERATIONS;
  float lr = argc > 1 ? atof(argv[2]) : LEARNING_RATE;
  float decay = argc > 1 ? atof(argv[3]) : WEIGHT_DECAY;

  /* load data */
  if (0 > load("../data/train-images-idx3-ubyte",
               16, X * DATAPOINTS, inputs))
    return -1;
  if (0 > load("../data/train-labels-idx1-ubyte",
               8, DATAPOINTS, labels))
    return -1;

  /* init weights */
  randn(w, .0f, 0.1f, X * H);
  randn(v, .0f, 0.1f, Y * H);

  double gflops_per_sample = (double)(2 * (X * H + H * Y) * 2) / (1 << 30);

  int samples = 0, epochs = 0;
  /* Fixed seed for reproducibility */
  srand(33);

  double t0 = get_time();
  double start_time = t0;

  /* TRAINING : Main Loop */
  do
  {
    /* FEED-FORWARD begin*/
    /* random sample */
    int r[B];

    for (int b = 0; b < B; b++)
      r[b] = random() % DATAPOINTS;

    memset(t, 0, sizeof(float) * Y * B);
    memset(h, 0, sizeof(float) * H * B);
    memset(y, 0, sizeof(float) * Y * B);

    for (int b = 0; b < B; b++)
    {
      t[b * Y + labels[r[b]]] = 1.0f;
      for (int i = 0; i < X; i++)
        x[b * X + i] = inputs[r[b] * X + i] / 255.0f;
    }

    /* h := w'x */
    /* col major */
    /* h [H rows, B cols] */
    /* w [X rows, H cols] */
    /* x [X rows, B cols] */
#pragma omp parallel for // collapse(3)
    for (int j = 0; j < H; j++)
      for (int i = 0; i < X; i++)
        for (int b = 0; b < B; b++)
          h[b * H + j] +=
              w[j * X + i] * x[b * X + i];

    /* activation function (nonlinearity) */
    for (int j = 0; j < H * B; j++)
#if LOGISTIC
      h[j] = 1.0f / (1.0f + expf(-h[j]));
#endif
#if RELU
    h[j] = h[j] < 0.0f ? 0.0f : h[j];
#endif
#if TANH
    h[j] = tanhf(h[j]);
#endif

    /* dropout if set*/
    if (DROPOUT > 0)
    {
      for (int j = 0; j < H * B; j++)
      {
        m[j] = ((float)random() / (float)RAND_MAX) < DROPOUT ? 0.0f : 1.0f;
        h[j] *= m[j];
      }
    }

    float act_sum = 0.0f;
    for (int i = 0; i < H * B; i++)
      act_sum += h[i];

    smooth_act = SMOOTHING * smooth_act + (1.0f - SMOOTHING) * act_sum / (H * B);

/* y := vh */
/* col major */
#pragma omp parallel for // collapse(3)
    for (int b = 0; b < B; b++)
      for (int j = 0; j < H; j++)
        for (int k = 0; k < Y; k++)
          y[b * Y + k] += v[k * H + j] * h[b * H + j];

    /* p := softmax(y) */
    for (int b = 0; b < B; b++)
    {
      float m0 = .0f; /* find max */
      for (int k = 0; k < Y; k++)
        m0 = k > 0 && (y[b * Y + k] > m0) ? y[b * Y + k] : m0;

      float sum = .0f;

      for (int k = 0; k < Y; k++)
      {
        p[b * Y + k] = expf(y[b * Y + k] - m0);
        sum += p[b * Y + k];
      }

      for (int k = 0; k < Y; k++)
        p[b * Y + k] /= sum;
    }

    /* FEED-FORWARD end */

    /* Computing stats */
    int argmax[B];
    float probmax[B];

    /* Compute the argmax for each batch*/
    for (int b = 0; b < B; b++)
    {
      argmax[b] = -1;
      probmax[b] = .0f;
      for (int k = 0; k < Y; k++)
      {
        if (probmax[b] < p[b * Y + k] || k == 0)
        {
          probmax[b] = p[b * Y + k];
          argmax[b] = k;
        }
        c[b * Y + k] = -logf(p[b * Y + k]) * t[b * Y + k];
        smooth_ce = smooth_ce * SMOOTHING +
                    (1.0f - SMOOTHING) * c[b * Y + k];
      }
      smooth_acc = smooth_acc * SMOOTHING +
                   (1.0f - SMOOTHING) * (argmax[b] == labels[r[b]]);
    }

    if (0 == (samples % STATS_INTERVAL) && samples > 0)
    {
      float time_d = get_time() - t0;
      float samples_per_sec = STATS_INTERVAL / time_d;
      float gflops_per_sec = samples_per_sec *
                             gflops_per_sample;
      printf("[%4.3f s] "
             "acc=%3.2f%%, "
             "ce=%3.3f, "
             "%.2f samples/sec, "
             "%.2f gflop/s\n",
             get_time() - start_time, 100.0 * smooth_acc, smooth_ce,
             samples_per_sec, gflops_per_sec);

      t0 = get_time();
    }

    /* backprop begin */
    /* reset grads */
    // cuda_check_error(cudaMemset(dh, 0, sizeof(float) * H * B));
    // cuda_check_error(cudaMemset(dw, 0, sizeof(float) * H * X));
    // cuda_check_error(cudaMemset(dv, 0, sizeof(float) * H * Y));

    /* dy */
    // for (int b = 0; b < B; b++)
    //   for (int k = 0; k < Y; k++)
    //     dy[b * Y + k] = p[b * Y + k] - t[b * Y + k];

    /* dv := h * dy' */
    // for (int b = 0; b < B; b++)
    //   for (int j = 0; j < H; j++)
    //     for (int k = 0; k < Y; k++)
    //       dv[k * H + j] += h[b * H + j] * dy[b * Y + k];

    /* dh := v * dy */
    // for (int b = 0; b < B; b++)
    //   for (int j = 0; j < H; j++)
    //     for (int k = 0; k < Y; k++)
    //       dh[b * H + j] += v[k * H + j] * dy[b * Y + k];

    // #pragma omp parallel for
    // for (int b = 0; b < B; b++) {
    //     for (int j = 0; j < H; j++) {
    //         for (int k = 0; k < Y; k++) {
    //             dy[b * Y + k] = p[b * Y + k] - t[b * Y + k];
    //             dv[k * H + j] += h[b * H + j] * dy[b * Y + k];
    //             dh[b * H + j] += v[k * H + j] * dy[b * Y + k];
    //         }
    //     }
    // }

    // Transfer memory from CPU to GPU
    // Launch kernel
    dimBlock = dim3(32, 32, 1);
    // dim3 dimGrid(Y / 8 + 1, H / 16 + 1, B / 8 + 1);
    dimGrid = dim3(Y / 32 + 1, H / 32 + 1, 1);
    backprop_kernel<<<dimGrid, dimBlock>>>(p, t, dv, v, dh, h);

    cuda_check_error(cudaGetLastError());

    cuda_check_error(cudaDeviceSynchronize());

    // Transfer memory from GPU to CPU

    /* nonlinearity on h */
    // #pragma omp parallel for
    //     for (int j = 0; j < H * B; j++)
    // #if LOGISTIC
    //       dh[j] = dh[j] * h[j] * (1.0f - h[j]);
    // #endif
    // #if RELU
    //     dh[j] = dh[j] * h[j];
    // #endif
    // #if TANH
    //     dh[j] = dh[j] * (1.0f - h[j] * h[j]);
    // #endif
    dimBlock = dim3(32, 32, 1);
    dimGrid = dim3(H / 32 + 1, B / 32 + 1, 1);
    nonlinearity_kernel<<<dimGrid, dimBlock>>>(dh, h);

    cuda_check_error(cudaGetLastError());

    cuda_check_error(cudaDeviceSynchronize());

    /* dw := x * dh' */
    // #pragma omp parallel for
    // for (int j = 0; j < H; j++)
    // {
    //   for (int i = 0; i < X; i++)
    //   {
    //     for (int b = 0; b < B; b++)
    //     {
    //       dw[j * X + i] += x[b * X + i] * dh[b * H + j];
    //     }
    //   }
    // }

    dimBlock = dim3(32, 32);
    dimGrid = dim3((X + 32 - 1) / 32, (H + 32 - 1) / 32);

    set_dw_zero_kernel<<<dimGrid, dimBlock>>>(dw);

    cuda_check_error(cudaGetLastError());

    cuda_check_error(cudaDeviceSynchronize());


    dimBlock = dim3(32, 32, 1);
    dimGrid = dim3(X / 32 + 1, H / 32 + 1, 1);

    update_weights_kernel<<<dimGrid, dimBlock>>>(dw, x, dh);

    cuda_check_error(cudaGetLastError());

    cuda_check_error(cudaDeviceSynchronize());

/* backprop end */

/* adjust weights */
#pragma omp parallel for
    for (int i = 0; i < H * X; i++)
    {
      w[i] = w[i] * (1.0f - decay) - dw[i] * lr;
    }
#pragma omp parallel for
    for (int i = 0; i < H * Y; i++)
    {
      v[i] = v[i] * (1.0f - decay) - dv[i] * lr;
    }

    samples += B;

  } while (epochs++ < max_iters && smooth_acc < TARGET_ACC);

  // Print total time
  printf("Total time: %f\n", get_time() - start_time);

  /* cleanup - Main */
  free(x), free(w), free(dw);
  free(m);
  free(y);
  free(c);

  /* cleanup - GPU */
  cudaFree(p);
  cudaFree(t);
  cudaFree(h);
  cudaFree(v);
  cudaFree(dy);
  cudaFree(dv);
  cudaFree(dh);

  return 0;
}
