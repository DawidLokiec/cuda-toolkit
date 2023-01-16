# My CUDA Toolkit

This CMake-based project contains some wrappers around the CUDA functions I use frequently. 
The wrappers are mainly concerned with throwing an exception with **meaningful** error messages in case of errors or **ensuring** that the GPU is always shut down properly and all alocated ressources are released. Some utility functions are also available. Occasionally this library will be expanded by me over time.

## API usage


Just include the header file `gpu_facade.cuh` into your code and use the global GPU facade object `CudaToolkit::gpuFacade`.

---
Feel free to use the repository or make interesting pull requests.
