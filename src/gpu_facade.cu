#include "gpu_facade.cuh"

using namespace CudaToolkit;

namespace {
	std::string getErrorDescription(const cudaError_t errorCode) {
		switch (errorCode) {
			case cudaErrorInvalidValue:
				return "'cudaErrorInvalidValue': "
					   "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.";
			case cudaErrorMemoryAllocation:
				return "'cudaErrorMemoryAllocation': "
					   "The API call failed because it was unable to allocate enough memory to perform the requested operation.";
			case cudaErrorInitializationError: // = 3
				return "'cudaErrorInitializationError'. "
					   "The API call failed because the CUDA driver and runtime could not be initialized.";
			case cudaErrorInvalidMemcpyDirection: // = 21
				return "'cudaErrorInvalidMemcpyDirection.' "
					   "This indicates that the direction of the memcpy passed to the API call is not one of the types specified by ::cudaMemcpyKind.";
			case cudaErrorInsufficientDriver: // = 35
				return "' cudaErrorInsufficientDriver': "
					   "This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. "
					   "Users should install an updated NVIDIA display driver to allow the application to run.";
			case cudaErrorNoDevice: // = 100
				return "'cudaErrorNoDevice': "
					   "This indicates that no CUDA-capable devices were detected by the installed CUDA driver.";
			case cudaErrorNotPermitted: // = 800
				return "'cudaErrorNotPermitted': "
					   "This error indicates the attempted operation is not permitted.";
			default:
				return "'" + std::to_string(errorCode) + "': Unknown error.";
		}
	}
}


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

void GpuFacade::copyDataBetweenCpuAndGpu(
		void *destinationGpuMemory,
		const void *sourceCpuMemory,
		const size_t numBytesToCopy,
		const cudaMemcpyKind copyDirection
) {
	const cudaError_t status = cudaMemcpy(
			destinationGpuMemory,
			sourceCpuMemory,
			numBytesToCopy,
			copyDirection
	);
	if (status) {
		throw std::runtime_error("cudaMemcpy call failed with error code " + getErrorDescription(status));
	} else {
		gpuUsedByCurrentProcess = true;
	}
}
