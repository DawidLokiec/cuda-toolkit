#ifndef CUDA_TOOLKIT_UTILS_CUH
#define CUDA_TOOLKIT_UTILS_CUH

#include <cstddef>
#include <driver_types.h>

namespace CudaUtils {

	/**
	 * @brief Allocates memory on the GPU.
	 * @details Allocates allocationSizeInBytes bytes of linear memory on the GPU and returns in *pointerToAllocatedMemory
	 * a pointer to the allocated memory. The allocated memory is suitably aligned for any kind of variable. The memory is
	 * not cleared.
	 * @param pointerToAllocatedMemory pointer to the allocated GPU memory.
	 * @param memorySizeInBytes the memory size in bytes to allocate.
	 */
	[[maybe_unused]] __host__ void allocateGpuMemory(void **pointerToAllocatedMemory, size_t memorySizeInBytes);

	/**
	 * @brief Frees memory on the GPU.
	 * @details Frees the memory space pointed to by gpuMemoryPointer, which must have been returned by a previous call
	 * of the memory allocation function allocateGpuMemory. Callers must ensure that all accesses to the pointer have
	 * completed before invoking freeGpuMemory. If freeGpuMemory(gpuMemoryPointer) has already been called before,
	 * an exception is thrown. If gpuMemoryPointer is nullptr, no operation is performed.
	 * @param gpuMemoryPointer the pointer to GPU memory to free.
	 */
	[[maybe_unused]] __host__ void freeGpuMemory(void **gpuMemoryPointer);

	/**
	 * @brief Copies data from CPU memory to GPU memory.
	 * @details Copies numBytesToCopy bytes from the memory area pointed to by sourceCpuMemory to the memory area
	 * pointed to by destinationGpuMemory.
	 * @param sourceCpuMemory the source CPU memory address.
	 * @param destinationGpuMemory the destination GPU memory address.
	 * @param numBytesToCopy the size in bytes to copy.
	 */
	[[maybe_unused]] __host__ void copyDataFromCpuMemoryToGpuMemory(
			const void *sourceCpuMemory,
			void *destinationGpuMemory,
			size_t numBytesToCopy
	);

	/**
 	 * @brief Copies data from GPU memory to CPU memory.
 	 * @details Copies numBytesToCopy bytes from the memory area pointed to by sourceGpuMemory to the memory area
	 * pointed to by destinationCpuMemory.
 	 * @param sourceGpuMemory the source GPU memory address.
 	 * @param destinationGpuMemory the destination CPU memory address.
     * @param numBytesToCopy the size in bytes to copy.
 	 */
	[[maybe_unused]] __host__ void copyDataFromGpuMemoryToCpuMemory(
			const void *sourceGpuMemory,
			void *destinationCpuMemory,
			size_t numBytesToCopy
	);
}

#endif //CUDA_TOOLKIT_UTILS_CUH
