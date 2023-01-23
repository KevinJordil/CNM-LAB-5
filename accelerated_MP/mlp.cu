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

/* batch size */
#define B 8
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
// #define RHO_TARGET 0.2f
// #define BETA 1e-5f

extern void randn(float *out, float mean, float std, int n);

unsigned char inputs[X * DATAPOINTS];
unsigned char labels[DATAPOINTS];

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
}

void cuda_check_error(cudaError err)
{
  if (err != cudaSuccess)
  {
    printf("CUDA error (%d): %s  \n", err, cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void myKernel(float *dy, float *dv, float *dh, float *p, float *t, float *h, float *v) {
    int b = blockIdx.x;
    int j = threadIdx.x;
    int k = threadIdx.y;
    dy[b * Y + k] = p[b * Y + k] - t[b * Y + k];
    dv[k * H + j] += h[b * H + j] * dy[b * Y + k];
    dh[b * H + j] += v[k * H + j] * dy[b * Y + k];
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


  // float *dy, float *dv, float *dh, float *p, float *t, float *h, float *v
  /* x -w-> h -v-> y */
  float *x, *h_h, *d_h, *y, *h_p, *d_p, *h_t, *d_t, *c; /*states*/
  float *w, *h_v, *d_v;                 /*weights*/
  float *h_dh, *d_dh, *h_dy, *d_dy;               /*states-grads*/
  float *dw, *h_dv, *d_dv;               /*weight-grads*/
  float *m;                     /*dropout*/

  /* allocate memory for arrays */
  x = (float *)malloc(sizeof(float) * X * B);
  w = (float *)malloc(sizeof(float) * X * H);
  dw = (float *)malloc(sizeof(float) * X * H);
  h_h = (float *)malloc(sizeof(float) * H * B);
  h_dh = (float *)malloc(sizeof(float) * H * B);
  m = (float *)malloc(sizeof(float) * H * B);
  h_v = (float *)malloc(sizeof(float) * H * Y);
  h_dv = (float *)malloc(sizeof(float) * Y * H);
  h_dy = (float *)malloc(sizeof(float) * Y * B);
  y = (float *)malloc(sizeof(float) * Y * B);
  h_p = (float *)malloc(sizeof(float) * Y * B);
  c = (float *)malloc(sizeof(float) * Y * B);
  h_t = (float *)malloc(sizeof(float) * Y * B);

  cuda_check_error(cudaMalloc((void **)&d_h, sizeof(float) * H * B));
  cuda_check_error(cudaMalloc((void **)&d_dh, sizeof(float) * H * B));
  cuda_check_error(cudaMalloc((void **)&d_v, sizeof(float) * H * Y));
  cuda_check_error(cudaMalloc((void **)&d_dv, sizeof(float) * Y * H));
  cuda_check_error(cudaMalloc((void **)&d_dy, sizeof(float) * Y * B));
  cuda_check_error(cudaMalloc((void **)&d_p, sizeof(float) * Y * B));
  cuda_check_error(cudaMalloc((void **)&d_t, sizeof(float) * Y * B));

  dim3 threadsPerBlock(H,Y);
  dim3 numBlocks(B);

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

  double gflops_per_sample =
      (double)(2 * (X * H + H * Y) * 2) /
      (1 << 30);

  int samples = 0, iters = 0;
  srand(33);

  double t0 = get_time();
  double start_time = t0;

  /* */
  do
  {
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
    #pragma omp parallel for// collapse(3)
    for (int j = 0; j < H; j++)
      for (int i = 0; i < X; i++)
        for (int b = 0; b < B; b++)
          h[b * H + j] +=
              w[j * X + i] * x[b * X + i];

    /* nonlinearity */
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

    /* forward pass end */
    /* bookkeeping for stats */
    int argmax[B];
    float probmax[B];
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

    if (0 == (samples % STATS_INTERVAL) &&
        samples > 0)
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
    memset(dh, 0, sizeof(float) * H * B);
    memset(dw, 0, sizeof(float) * H * X);
    memset(dv, 0, sizeof(float) * H * Y);

    /* dy */
    //for (int b = 0; b < B; b++)
    //  for (int k = 0; k < Y; k++)
    //    dy[b * Y + k] = p[b * Y + k] - t[b * Y + k];

    /* dv := h * dy' */
    //for (int b = 0; b < B; b++)
    //  for (int j = 0; j < H; j++)
    //    for (int k = 0; k < Y; k++)
    //      dv[k * H + j] += h[b * H + j] * dy[b * Y + k];

    /* dh := v * dy */
    //for (int b = 0; b < B; b++)
    //  for (int j = 0; j < H; j++)
    //    for (int k = 0; k < Y; k++)
    //      dh[b * H + j] += v[k * H + j] * dy[b * Y + k];

    //#pragma omp parallel for
    cuda_check_error(cudaMalloc((void **)&d_h, sizeof(float) * H * B));
    cuda_check_error(cudaMalloc((void **)&d_dh, sizeof(float) * H * B));
    cuda_check_error(cudaMalloc((void **)&d_v, sizeof(float) * H * Y));
    cuda_check_error(cudaMalloc((void **)&d_dv, sizeof(float) * Y * H));
    cuda_check_error(cudaMalloc((void **)&d_dy, sizeof(float) * Y * B));
    cuda_check_error(cudaMalloc((void **)&d_p, sizeof(float) * Y * B));
    cuda_check_error(cudaMalloc((void **)&d_t, sizeof(float) * Y * B));


    // Copy the memory from the host to the device
    cuda_check_error(cudaMemcpy(d_h, h_h, sizeof(float) * H * B, cudaMemcpyHostToDevice));
    cuda_check_error(cudaMemcpy(d_dh, h_dh, sizeof(float) * H * B, cudaMemcpyHostToDevice));
    cuda_check_error(cudaMemcpy(d_v, h_v, sizeof(float) * H * Y, cudaMemcpyHostToDevice));
    cuda_check_error(cudaMemcpy(d_dv, h_dv, sizeof(float) * Y * H, cudaMemcpyHostToDevice));
    cuda_check_error(cudaMemcpy(d_p, h_p, sizeof(float) * Y * B, cudaMemcpyHostToDevice));
    cuda_check_error(cudaMemcpy(d_t, h_t, sizeof(float) * Y * B, cudaMemcpyHostToDevice));

    // Launch the kernel
    myKernel<<<numBlocks, threadsPerBlock>>>(d_dy, d_dv, d_dh, d_p, d_t, d_h, d_v);

    // Copy the memory from the device to the host
    cuda_check_error(cudaMemcpy(h_dy, d_dy, sizeof(float) * Y * B, cudaMemcpyDeviceToHost));
    cuda_check_error(cudaMemcpy(h_dv, d_dv, sizeof(float) * Y * H, cudaMemcpyDeviceToHost));
    cuda_check_error(cudaMemcpy(h_dh, d_dh, sizeof(float) * H * B, cudaMemcpyDeviceToHost));

    /*
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < Y; k++) {
                dy[b * Y + k] = p[b * Y + k] - t[b * Y + k];
                dv[k * H + j] += h[b * H + j] * dy[b * Y + k];
                dh[b * H + j] += v[k * H + j] * dy[b * Y + k];
            }
        }
    }
    */

    /* nonlinearity on h */
    //#pragma omp parallel for
    for (int j = 0; j < H * B; j++)
#if LOGISTIC
      dh[j] = dh[j] * h[j] * (1.0f - h[j]);
#endif
#if RELU
    dh[j] = dh[j] * h[j];
#endif
#if TANH
    dh[j] = dh[j] * (1.0f - h[j] * h[j]);
#endif

    /* dw := x * dh' */
    #pragma omp parallel for
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < X; i++) {
            for (int b = 0; b < B; b++) {
                dw[j * X + i] += x[b * X + i] * dh[b * H + j];
            }
        }
    }
    /* backprop end */

    /* adjust weights */
    //#pragma omp parallel for
    for (int i = 0; i < H * X; i++) {
        w[i] = w[i] * (1.0f - decay) - dw[i] * lr;
    }
    //#pragma omp parallel for
    for (int i = 0; i < H * Y; i++) {
        v[i] = v[i] * (1.0f - decay) - dv[i] * lr;
    }

    samples += B;

  } while (iters++ < max_iters && smooth_acc < TARGET_ACC);

  /* cleanup */
  free(x), free(w), free(dw);
  free(h), free(dh);
  free(m);
  free(v), free(dv);
  free(y), free(dy);
  free(p), free(c), free(t);

  cuda_check_error(cudaFree(d_h));
  cuda_check_error(cudaFree(d_dh));
  cuda_check_error(cudaFree(d_v));
  cuda_check_error(cudaFree(d_dv));
  cuda_check_error(cudaFree(d_dy));
  cuda_check_error(cudaFree(d_p));
  cuda_check_error(cudaFree(d_t));



  return 0;
}
