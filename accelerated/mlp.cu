#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "io.h"

/* batch size */
#define B 8
/* input size */
#define X 784
/* hidden size */
#define H 100
/* output size */
#define Y 10

#define ITERATIONS 1000000
#define TARGET_ACC 0.95f
#define STATS_INTERVAL 50000
#define SMOOTHING 0.99999f
#define LEARNING_RATE 5 * 1e-4f
#define DATAPOINTS 50000
#define WEIGHT_DECAY .0f
#define LOGISTIC 0
#define DROPOUT 0.0
#define RELU 1
#define TANH 0

float randf() {
  return rand() / (RAND_MAX + 1.0f);
}

void randn(float *out, float mean, float std, int n) {
  for (int i=0; i<n; i++) {
    float  x = randf(),
           y = randf(),
           z = sqrtf(-2 * logf(x)) * cos(2 * M_PI * y);
    out[i] = std*z + mean;
  }
}

unsigned char inputs[X * DATAPOINTS];
unsigned char labels[DATAPOINTS];

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
}

__global__ void forward_kernel(float *x, float *w, float *h, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    for (int j = 0; j < H; j++)
    {
      h[i * H + j] = 0;
      for (int k = 0; k < X; k++)
      {
        h[i * H + j] += x[i * X + k] * w[k * H + j];
      }
      h[i * H + j] = h[i * H + j] > 0. ? h[i * H + j] : 0.;
    }
  }
}

__global__ void backward_kernel(float *dh, float *w, float *x, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    for (int j = 0; j < X; j++)
    {
      x[i * X + j] = 0;
      for (int k = 0; k < H; k++)
      {
        x[i * X + j] += dh[i * H + k] * w[j * H + k];
      }
    }
  }
}

__global__ void update_kernel(float *w, float *dw, float *x, float *dh, float lr, float decay, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    for (int j = 0; j < H; j++)
    {
      for (int k = 0; k < X; k++)
      {
        dw[k * H + j] = lr * (x[i * X + k] * dh[i * H + j] - decay * w[k * H + j]);
        w[k * H + j] += dw[k * H + j];
      }
    }
  }
}

int main(int argc, char **argv)
{
  /* x -w-> h -v-> y */
  float *x, *h, *y, *p, *t, *c; /*states*/
  float *w, *v;                 /*weights*/
  float *dh, *dy;               /*states-grads*/
  float *dw, *dv;               /*weight-grads*/
  float *m;                     /*dropout*/

  x = (float *)malloc(sizeof(float) * X * B);
  w = (float *)malloc(sizeof(float) * X * H);
  dw = (float *)malloc(sizeof(float) * X * H);
  h = (float *)malloc(sizeof(float) * H * B);
  dh = (float *)malloc(sizeof(float) * H * B);
  m = (float *)malloc(sizeof(float) * H * B);
  v = (float *)malloc(sizeof(float) * H * Y);
  dv = (float *)malloc(sizeof(float) * Y * H);
  dy = (float *)malloc(sizeof(float) * Y * B);
  y = (float *)malloc(sizeof(float) * Y * B);
  p = (float *)malloc(sizeof(float) * Y * B);
  c = (float *)malloc(sizeof(float) * Y * B);
  t = (float *)malloc(sizeof(float) * Y * B);

  //float smooth_act = 0.0f;
  float smooth_ce = logf(Y);
  //float smooth_acc = 1.0f / Y;

  int max_iters = argc > 1 ? atoi(argv[1]) : ITERATIONS;
  float lr = argc > 1 ? atof(argv[2]) : LEARNING_RATE;
  float decay = argc > 1 ? atof(argv[3]) : WEIGHT_DECAY;

  if (0 > load("../data/train-images-idx3-ubyte",
               16, X * DATAPOINTS, inputs))
    return -1;
  if (0 > load("../data/train-labels-idx1-ubyte",
               8, DATAPOINTS, labels))
    return -1;

  /* init weights */
  randn(w, .0f, 0.1f, X * H);
  randn(v, .0f, 0.1f, Y * H);

  /* set up CUDA */
  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasHandle_t handle;
  cublasCreate(&handle);

  /* transfer data to GPU */
  float *d_x, *d_w, *d_dw, *d_h, *d_dh, *d_m;
  cudaMalloc(&d_x, sizeof(float) * X * B);
  cudaMalloc(&d_w, sizeof(float) * X * H);
  cudaMalloc(&d_dw, sizeof(float) * X * H);
  cudaMalloc(&d_h, sizeof(float) * H * B);
  cudaMalloc(&d_dh, sizeof(float) * H * B);
  cudaMalloc(&d_m, sizeof(float) * H * B);
  cudaMemcpyAsync(d_x, x, sizeof(float) * X * B, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_w, w, sizeof(float) * X * H, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_dw, dw, sizeof(float) * X * H, cudaMemcpyHostToDevice, stream);

  
  /* training loop */
  for (int i = 0; i < max_iters; i++)
  {

    /* forward pass */
    forward_kernel<<<B, H>>>(d_x, d_w, d_h, B);

    /* backward pass */
    backward_kernel<<<B, X>>>(d_dh, d_w, d_x, B);

    /* update weights */
    update_kernel<<<B, X>>>(d_w, d_dw, d_x, d_dh, lr, decay, B);

    /* transfer data back to host */
    cudaMemcpyAsync(x, d_x, sizeof(float) * X * B, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(w, d_w, sizeof(float) * X * H, cudaMemcpyDeviceToHost, stream);
  }

  /* clean up */
  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_dw);
  cudaFree(d_h);
  cudaFree(d_dh);
  cudaFree(d_m);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  free(x);
  free(w);
  free(dw);
  free(h);
  free(dh);
  free(m);
  free(v);
  free(dv);
  free(dy);
  free(y);
  free(p);
  free(c);
  free(t);
  return 0;
}
