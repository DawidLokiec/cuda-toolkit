#include "error.cuh"

std::string CudaUtils::toErrorDescription(const cudaError_t cudaErrorCode) {
	switch (cudaErrorCode) {
		case cudaSuccess:
			return "'cudaSuccess'. The API call returned with no errors.";
		case cudaErrorInvalidValue:
			return "'cudaErrorInvalidValue'. "
				   "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.";
		case cudaErrorMemoryAllocation:
			return "'cudaErrorMemoryAllocation'. "
				   "The API call failed because it was unable to allocate enough memory to perform the requested operation.";
		case cudaErrorInitializationError: // = 3
			return "'cudaErrorInitializationError'. "
				   "The API call failed because the CUDA driver and runtime could not be initialized.";
		case cudaErrorInvalidMemcpyDirection: // = 21
			return "'cudaErrorInvalidMemcpyDirection.' "
				   "This indicates that the direction of the memcpy passed to the API call is not one of the types specified by ::cudaMemcpyKind.";
		case cudaErrorInsufficientDriver: // = 35
			return "' cudaErrorInsufficientDriver'. "
				   "This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. "
				   "Users should install an updated NVIDIA display driver to allow the application to run.";
		case cudaErrorNoDevice: // = 100
			return "'cudaErrorNoDevice'. "
				   "This indicates that no CUDA-capable devices were detected by the installed CUDA driver.";
		case cudaErrorNotPermitted: // = 800
			return "'cudaErrorNotPermitted. '"
				   "This error indicates the attempted operation is not permitted.";
		default:
			return "Unknown error code: " + std::to_string(cudaErrorCode);
	}
}