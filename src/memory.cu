#include "memory.cuh"
#include "error.cuh"
#include <stdexcept>

namespace {
	__host__ void copyDataBetweenCpuAndGpu(
			void *destinationGpuMemory,
			const void *sourceCpuMemory,
			const size_t numBytesToCopy,
			const cudaMemcpyKind copyDirection
	) {
		const cudaError_t status = cudaMemcpy(destinationGpuMemory, sourceCpuMemory, numBytesToCopy, copyDirection);
		if (status != cudaSuccess) {
			if (copyDirection == cudaMemcpyHostToDevice) {
				throw std::runtime_error(
						"copyDataFromCpuMemoryToGpuMemory call failed with the CUDA error code " +
						CudaUtils::toErrorDescription(status)
				);
			} else {
				throw std::runtime_error(
						"copyDataFromGpuMemoryToCpuMemory call failed with the CUDA error code " +
						CudaUtils::toErrorDescription(status)
				);
			}
		}
	}
}

[[maybe_unused]] __host__ void CudaUtils::allocateGpuMemory(
		void **pointerToAllocatedMemory,
		const size_t memorySizeInBytes
) {
	const cudaError_t status = cudaMalloc(pointerToAllocatedMemory, memorySizeInBytes);
	if (status != cudaSuccess) {
		throw std::runtime_error(
				"allocateGpuMemory call failed with the CUDA error code " + CudaUtils::toErrorDescription(status)
		);
	}
}

[[maybe_unused]] __host__ void CudaUtils::freeGpuMemory(void **gpuMemoryPointer) {
	const cudaError_t status = cudaFree(gpuMemoryPointer);
	if (status != cudaSuccess) {
		throw std::runtime_error(
				"freeGpuMemory call failed with the CUDA error code " + CudaUtils::toErrorDescription(status)
		);
	}

}

[[maybe_unused]] __host__ void CudaUtils::copyDataFromCpuMemoryToGpuMemory(
		const void *sourceCpuMemory,
		void *destinationGpuMemory,
		const size_t numBytesToCopy
) {
	copyDataBetweenCpuAndGpu(destinationGpuMemory, sourceCpuMemory, numBytesToCopy, cudaMemcpyHostToDevice);
}

[[maybe_unused]] __host__ void CudaUtils::copyDataFromGpuMemoryToCpuMemory(
		const void *sourceGpuMemory,
		void *destinationCpuMemory,
		const size_t numBytesToCopy
) {
	copyDataBetweenCpuAndGpu(destinationCpuMemory, sourceGpuMemory, numBytesToCopy, cudaMemcpyDeviceToHost);
}