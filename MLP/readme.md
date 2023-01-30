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

```
cnm@cnmlab:~/CNM-MLP-C/MLP/accelerated_only_MP$ ./mlp
[46.092 s] acc=40.99%, ce=0.045, 1084.79 samples/sec, 3.21 gflop/s
[92.241 s] acc=61.63%, ce=0.022, 1083.43 samples/sec, 3.20 gflop/s
[138.231 s] acc=74.50%, ce=0.019, 1087.20 samples/sec, 3.22 gflop/s
[184.198 s] acc=82.59%, ce=0.016, 1087.76 samples/sec, 3.22 gflop/s
[230.153 s] acc=87.68%, ce=0.015, 1088.01 samples/sec, 3.22 gflop/s
[276.135 s] acc=90.91%, ce=0.013, 1087.40 samples/sec, 3.22 gflop/s
[322.124 s] acc=93.01%, ce=0.013, 1087.23 samples/sec, 3.22 gflop/s
[368.109 s] acc=94.43%, ce=0.012, 1087.32 samples/sec, 3.22 gflop/s
Total time: 393.993599
```


## Accelerated OpenMP and CUDA

```
cnm@cnmlab:~/CNM-MLP-C/MLP/accelerated$ sudo /usr/local/cuda/bin/nvprof ./mlp
[sudo] password for cnm:
==27977== NVPROF is profiling process 27977, command: ./mlp
==27977== Warning: Unified Memory Profiling is not supported on the underlying platform. System requirements for unified memory can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
[64.577 s] acc=36.72%, ce=0.061, 774.27 samples/sec, 2.29 gflop/s
[130.425 s] acc=57.61%, ce=0.034, 759.32 samples/sec, 2.25 gflop/s
[195.328 s] acc=70.85%, ce=0.029, 770.39 samples/sec, 2.28 gflop/s
[261.072 s] acc=79.38%, ce=0.025, 760.52 samples/sec, 2.25 gflop/s
[326.487 s] acc=84.74%, ce=0.024, 764.35 samples/sec, 2.26 gflop/s
[391.392 s] acc=88.17%, ce=0.021, 770.37 samples/sec, 2.28 gflop/s
[455.994 s] acc=90.38%, ce=0.021, 773.96 samples/sec, 2.29 gflop/s
[519.951 s] acc=91.87%, ce=0.019, 781.79 samples/sec, 2.31 gflop/s
[584.651 s] acc=92.86%, ce=0.020, 772.79 samples/sec, 2.29 gflop/s
[649.924 s] acc=93.56%, ce=0.018, 766.02 samples/sec, 2.27 gflop/s
[714.872 s] acc=94.03%, ce=0.018, 769.84 samples/sec, 2.28 gflop/s
[780.575 s] acc=94.48%, ce=0.017, 761.00 samples/sec, 2.25 gflop/s
[845.612 s] acc=94.74%, ce=0.016, 768.80 samples/sec, 2.27 gflop/s
Total time: 891.121454
==27977== Profiling application: ./mlp
==27977== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.10%  15.4163s    410754  37.531us  2.9160us  991.89us  [CUDA memcpy HtoH]
                   35.72%  11.6917s     68459  170.78us  166.57us  175.52us  backprop_kernel(float*, float*, float*, float*, float*, float*)
                   17.18%  5.62353s    136918  41.072us  40.989us  43.334us  [CUDA memset]
      API calls:   74.08%  145.662s    410754  354.62us  135.57us  119.92ms  cudaMemcpy
                   13.21%  25.9788s    136918  189.74us  149.90us  3.9103ms  cudaMemset
                    9.01%  17.7185s     68459  258.82us  73.803us  4.9247ms  cudaDeviceSynchronize
                    3.49%  6.86282s     68459  100.25us  89.429us  987.88us  cudaLaunchKernel
                    0.18%  344.80ms         7  49.257ms  92.450us  344.14ms  cudaMallocManaged
                    0.04%  72.091ms     68459  1.0530us     677ns  73.908us  cudaGetLastError
                    0.00%  953.77us         7  136.25us  105.84us  260.63us  cudaFree
                    0.00%  108.55us        97  1.1190us     572ns  26.876us  cuDeviceGetAttribute
                    0.00%  10.521us         1  10.521us  10.521us  10.521us  cuDeviceTotalMem
                    0.00%  7.3430us         3  2.4470us  1.5620us  3.6460us  cuDeviceGetCount
                    0.00%  3.8540us         2  1.9270us  1.3020us  2.5520us  cuDeviceGet
                    0.00%  1.5100us         1  1.5100us  1.5100us  1.5100us  cuDeviceGetName
                    0.00%     938ns         1     938ns     938ns     938ns  cuDeviceGetUuid
```