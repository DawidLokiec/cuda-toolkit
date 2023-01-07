#ifndef CUDA_TOOLKIT_ERROR_CUH
#define CUDA_TOOLKIT_ERROR_CUH

#include <string>

namespace CudaUtils {

	/**
	 * @brief Converts the passed CUDA error code into a description of the possible error cause.
	 * @param cudaErrorCode the error code to be converted into a meaningful error description.
	 * @return A description of the meaning of the error code passed.
	 */
	std::string toErrorDescription(cudaError_t cudaErrorCode);
}

#endif //CUDA_TOOLKIT_ERROR_CUH
