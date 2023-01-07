#include "gpu_facade.cuh"
#include "memory.cuh"

using namespace CudaToolkit;

GpuFacade::GpuFacade() : gpuUsedByCurrentProcess(false) {

}

GpuFacade::~GpuFacade() {
	if (gpuUsedByCurrentProcess) {
		// Ensure the GPU is reset on program termination
		cudaDeviceReset();
		gpuUsedByCurrentProcess = false;
	}
}

[[maybe_unused]] GpuFacade &GpuFacade::GpuFacade::getInstance() {
	static GpuFacade instance;
	return instance;
}

[[maybe_unused]] __host__ void GpuFacade::allocateGpuMemory(
		void **pointerToAllocatedMemory,
		const size_t memorySizeInBytes
) {
	CudaUtils::allocateGpuMemory(pointerToAllocatedMemory, memorySizeInBytes);
	gpuUsedByCurrentProcess = true;
}


[[maybe_unused]] __host__ void GpuFacade::freeGpuMemory(void **gpuMemoryPointer) {
	CudaUtils::freeGpuMemory(gpuMemoryPointer);
	gpuUsedByCurrentProcess = true;
}

[[maybe_unused]] __host__ void GpuFacade::copyDataFromCpuMemoryToGpuMemory(
		const void *sourceCpuMemory,
		void *destinationGpuMemory,
		const size_t numBytesToCopy
) {
	CudaUtils::copyDataFromCpuMemoryToGpuMemory(sourceCpuMemory, destinationGpuMemory, numBytesToCopy);
	gpuUsedByCurrentProcess = true;
}

[[maybe_unused]] __host__ void GpuFacade::copyDataFromGpuMemoryToCpuMemory(
		const void *sourceGpuMemory,
		void *destinationCpuMemory,
		const size_t numBytesToCopy
) {
	CudaUtils::copyDataFromGpuMemoryToCpuMemory(sourceGpuMemory, destinationCpuMemory, numBytesToCopy);
	gpuUsedByCurrentProcess = true;
}
