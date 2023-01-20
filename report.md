## Description of application.

1 page

### Algorithm

mlp

### Dataset

## Application bottleneck

1 page

- Execution time: what is the execution time of the application?
- What is the complexity of the application (constant O(1), linear O(n), other?)
- What are the bottlenecks?
- Which parts of the application do you plan to accelerate and why?
- What is theoretically the performance that you could and would like to achieve?
  (make the link with the theory of the course about performances, lesson of 9.1.23)

## Acceleration

2 pages

Given the analysis of stage 2, we would like you to pick a technology for acceleration and describe your
choices.

- Describe how you plan to accelerate your application (CPU vs GPU, using OpenMP, CUDA
  libraries). At least you will need to accelerate one part of the application on GPU, using a library
  of your choice. Feel free to use also OpenMP on the CPU.
- Which libraries are you going to use for GPU acceleration? You will most likely need to use
  either `cuBLAS` or `cuDNN`. We will provide you with manuals for both libraries so that you can
  follow the steps.
- Analyse the memory needs of your application before starting: how big is the data? Where is
  it? What is the expected data locality?

## Analysis of results

1 page describing the results of the acceleration.
1 page potential future lines.

Given the work you performed in Stage 3, we want you know to analyse the results obtained and draw
some conclusions:

- Analysis of the results: what is the performance enhancement achieved? Where does it come
  from and why? What is the new bottleneck created?
- Potential future lines: seeing the acceleration results, what other things should you accelerate?
  Now that you see the results, should you have done something different?
