#ifndef CUDA_TOOLKIT_GPU_CODE_UTILITIES_CUH
#define CUDA_TOOLKIT_GPU_CODE_UTILITIES_CUH

/**
 * The namespace of this library.
 */
namespace CudaToolkit::GpuCodeUtilities {

	[[maybe_unused]] __device__ inline unsigned int getGlobalThreadId() {
		return blockIdx.x * blockDim.x + threadIdx.x;
	}

	[[maybe_unused]] __device__ float castDoubleToFloat(double value);
}

#endif //CUDA_TOOLKIT_GPU_CODE_UTILITIES_CUH
