# Results

## Original

```
[67.270 s] acc=40.99%, ce=0.045, 743.28 samples/sec, 2.20 gflop/s
[134.516 s] acc=61.63%, ce=0.022, 743.53 samples/sec, 2.20 gflop/s
[201.754 s] acc=74.50%, ce=0.019, 743.64 samples/sec, 2.20 gflop/s
[268.999 s] acc=82.60%, ce=0.016, 743.54 samples/sec, 2.20 gflop/s
[336.225 s] acc=87.68%, ce=0.015, 743.77 samples/sec, 2.20 gflop/s
[403.459 s] acc=90.91%, ce=0.013, 743.67 samples/sec, 2.20 gflop/s
[470.688 s] acc=93.01%, ce=0.013, 743.73 samples/sec, 2.20 gflop/s
[537.911 s] acc=94.43%, ce=0.012, 743.79 samples/sec, 2.20 gflop/s
Total time: 575.676547
```

## Accelerated OpenMP only

TODO


## Accelerated OpenMP and CUDA

```
cnm@cnmlab:~/CNM-MLP-C/MLP/accelerated$ sudo /usr/local/cuda/bin/nvprof ./mlp
[sudo] password for cnm:
==23248== NVPROF is profiling process 23248, command: ./mlp
==23248== Warning: Unified Memory Profiling is not supported on the underlying platform. System requirements for unified memory can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
[58.031 s] acc=36.72%, ce=0.061, 861.60 samples/sec, 2.55 gflop/s
[116.665 s] acc=57.61%, ce=0.034, 852.75 samples/sec, 2.52 gflop/s
[174.413 s] acc=70.85%, ce=0.029, 865.83 samples/sec, 2.56 gflop/s
[232.808 s] acc=79.37%, ce=0.025, 856.24 samples/sec, 2.53 gflop/s
[291.408 s] acc=84.73%, ce=0.024, 853.25 samples/sec, 2.52 gflop/s
[348.941 s] acc=88.17%, ce=0.021, 869.07 samples/sec, 2.57 gflop/s
[407.068 s] acc=90.38%, ce=0.021, 860.18 samples/sec, 2.54 gflop/s
[463.568 s] acc=91.87%, ce=0.019, 884.95 samples/sec, 2.62 gflop/s
[520.667 s] acc=92.86%, ce=0.020, 875.68 samples/sec, 2.59 gflop/s
[577.546 s] acc=93.56%, ce=0.018, 879.06 samples/sec, 2.60 gflop/s
[634.099 s] acc=94.03%, ce=0.018, 884.12 samples/sec, 2.62 gflop/s
[691.713 s] acc=94.48%, ce=0.017, 867.84 samples/sec, 2.57 gflop/s
[748.778 s] acc=94.74%, ce=0.016, 876.20 samples/sec, 2.59 gflop/s
==23248== Profiling application: ./mlp
==23248== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.10%  15.4144s    410754  37.526us  2.9160us  989.29us  [CUDA memcpy HtoH]
                   35.72%  11.6899s     68459  170.76us  166.36us  175.16us  backprop_kernel(float*, float*, float*, float*, float*, float*)
                   17.18%  5.62428s    136918  41.077us  40.989us  43.335us  [CUDA memset]
      API calls:   74.06%  144.327s    410754  351.37us  130.84us  119.38ms  cudaMemcpy
                   13.25%  25.8131s    136918  188.53us  151.10us  2.8159ms  cudaMemset
                    9.04%  17.6243s     68459  257.44us  73.490us  1.1777ms  cudaDeviceSynchronize
                    3.44%  6.69693s     68459  97.823us  87.605us  975.48us  cudaLaunchKernel
                    0.17%  332.98ms         7  47.569ms  87.709us  332.35ms  cudaMallocManaged
                    0.04%  76.176ms     68459  1.1120us     677ns  49.740us  cudaGetLastError
                    0.00%  936.83us         7  133.83us  104.22us  257.61us  cudaFree
                    0.00%  127.87us        97  1.3180us     625ns  27.761us  cuDeviceGetAttribute
                    0.00%  13.854us         1  13.854us  13.854us  13.854us  cuDeviceTotalMem
                    0.00%  6.7190us         3  2.2390us  1.3540us  3.3340us  cuDeviceGetCount
                    0.00%  3.3850us         2  1.6920us  1.3020us  2.0830us  cuDeviceGet
                    0.00%  2.0320us         1  2.0320us  2.0320us  2.0320us  cuDeviceGetName
                    0.00%     990ns         1     990ns     990ns     990ns  cuDeviceGetUuid
```