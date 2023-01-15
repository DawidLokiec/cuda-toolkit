#ifndef CUDA_TOOLKIT_GPU_FACADE_CUH
#define CUDA_TOOLKIT_GPU_FACADE_CUH

#include "gpu_memory.cuh"

/**
 * The namespace of this current toolkit.
 */
namespace CudaToolkit {

	/**
 	 * @brief Represents a facade to the CUDA-able GPU.
 	 * @details After the program termination the used GPU is reset by this class automatically. Therefore the caller
 	 * no need to call cudaDeviceReset() explicitly.
 	 */
	class GpuFacade {
		private:
			/**
			 * The boolean flag whether the GPU is being used by the current process.
			 */
			volatile bool gpuUsedByCurrentProcess;

			/**
			 * The default constructor. Creates a new instance of the current class.
			 */
			GpuFacade();

		public:
			/**
			 * The copy constructor.
			  */
			GpuFacade(GpuFacade const &) = delete;

			/**
			 * The assignment operator.
			 */
			void operator=(GpuFacade const &) = delete;

			/**
			 * The destructor. Resets all CUDA-able GPUs used by the current process.
			 */
			~GpuFacade();

			/**
			 * Returns the singleton instance of this class.
			 * @return the singleton instance of this class.
			 */
			[[maybe_unused]] __host__ static GpuFacade &getInstance();

			/**
			 * @brief Allocates memory on the GPU.
			 * @details Allocates memorySizeInBytes bytes of linear memory on the GPU. The allocated memory is suitably
			 * aligned for any kind of variable. The memory is freed by the destructor of the returned object.
			 * @param memorySizeInBytes the memory size in bytes to allocate.
			 */
			template<typename T>
			[[maybe_unused]] __host__ GpuMemory<T> allocateGpuMemory(const size_t memorySizeInBytes) {
				auto tmp = GpuMemory<T>::allocate(memorySizeInBytes);
				gpuUsedByCurrentProcess = true;
				return tmp;
			}

			/**
			 * @brief Copies data from CPU memory to GPU memory.
			 * @details Copies numBytesToCopy bytes from the memory area pointed to by sourceCpuMemory to the memory
			 * area pointed to by destinationGpuMemory.
			 * @param sourceCpuMemory the source CPU memory address.
			 * @param destinationGpuMemory the destination GPU memory.
			 * @param numBytesToCopy the size in bytes to copy.
			 */
			template<typename T>
			[[maybe_unused]] __host__ void copyDataFromCpuMemoryToGpuMemory(
					const void *sourceCpuMemory,
					const GpuMemory<T> &destinationGpuMemory,
					size_t numBytesToCopy
			) {
				copyDataBetweenCpuAndGpu(
						destinationGpuMemory,
						sourceCpuMemory,
						numBytesToCopy,
						cudaMemcpyHostToDevice
				);
			}

			/**
			  * @brief Copies data from GPU memory to CPU memory.
			  * @details Copies numBytesToCopy bytes from the memory area pointed to by sourceGpuMemory to the memory
			  * area pointed to by destinationCpuMemory.
			  * @param sourceGpuMemory the source GPU memory.
			  * @param destinationGpuMemory the destination CPU memory address.
			  * @param numBytesToCopy the size in bytes to copy.
			  */
			template<typename T>
			[[maybe_unused]] __host__ void copyDataFromGpuMemoryToCpuMemory(
					const GpuMemory<T> &sourceGpuMemory,
					void *destinationCpuMemory,
					size_t numBytesToCopy
			) {
				copyDataBetweenCpuAndGpu(
						destinationCpuMemory,
						sourceGpuMemory,
						numBytesToCopy,
						cudaMemcpyDeviceToHost
				);
			}

		private:
			__host__ void copyDataBetweenCpuAndGpu(
					void *destinationGpuMemory,
					const void *sourceCpuMemory,
					size_t numBytesToCopy,
					cudaMemcpyKind copyDirection
			);
	};

	[[maybe_unused]] inline GpuFacade &gpuFacade = GpuFacade::getInstance();
}

#endif //CUDA_TOOLKIT_GPU_FACADE_CUH
