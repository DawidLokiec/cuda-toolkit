#include "gpu_facade.cuh"

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

[[maybe_unused]] GpuFacade& gpuFacade = GpuFacade::getInstance();