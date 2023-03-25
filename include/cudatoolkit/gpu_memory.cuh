#ifndef CUDA_TOOLKIT_GPU_MEMORY_H
#define CUDA_TOOLKIT_GPU_MEMORY_H

#include <stdexcept>
#include <iostream>
#include <string>

/**
 * The namespace of this toolkit.
 */
namespace CudaToolkit {

	/**
	 * @brief Represents memory space on the GPU.
	 * @tparam T the data type of the data stored in the memory space.
	 */
	template<class T>
	class GpuMemory {

		private:
			/**
			 * @brief The pointer to the current GPU memory.
			 */
			T *pMemory_;

			/**
			 * @brief The size of the current GPU memory in bytes.
			 */
			size_t memorySizeInBytes_;

			__host__ explicit GpuMemory(const size_t memorySizeInBytes) :
					pMemory_(nullptr),
					memorySizeInBytes_(memorySizeInBytes) {
				const cudaError_t status = cudaMalloc(&pMemory_, memorySizeInBytes);
				if (status) {
					// let it crash
					throw std::runtime_error("cudaMalloc call failed with error code " + getErrorDescription(status));
				}
			}

		public:
			[[maybe_unused]] __host__ static GpuMemory allocate(const size_t memorySizeInBytes) { // clean code
				return GpuMemory<T>(memorySizeInBytes);
			}

			~GpuMemory() {
				const cudaError_t status = cudaFree(pMemory_);
				if (status) {
					std::cerr << "cudaFree call failed with error code " << getErrorDescription(status) << std::endl;
				}
			}

			[[maybe_unused]] __host__ size_t getMemorySizeInBytes() const {
				return memorySizeInBytes_;
			}

			__host__ operator void *() const { // NOLINT(google-explicit-constructor)
				return pMemory_;
			}

			__host__ operator T *() const { // NOLINT(google-explicit-constructor)
				return pMemory_;
			}

		private:
			__host__ static std::string getErrorDescription(const cudaError_t errorCode) {
				switch (errorCode) {
					case cudaSuccess: // = 0
						return "'cudaSuccess': No errors occurred.";
					case cudaErrorInvalidValue: // = 1
						return "'cudaErrorInvalidValue': One or more of the parameters passed to the API call is not "
							   "within an acceptable range of values.";
					case cudaErrorMemoryAllocation: // = 2
						return "'cudaErrorMemoryAllocation': Unable to allocate enough memory to perform the requested "
							   "operation.";
					default:
						return "'" + std::to_string(errorCode) + "': Unknown error code.";
				}
			}
	};
}

#endif //CUDA_TOOLKIT_GPU_MEMORY_H
