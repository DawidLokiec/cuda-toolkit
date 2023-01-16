#include "gpu_code_utilities.cuh"

using namespace CudaToolkit;

[[maybe_unused]] __device__ float GpuCodeUtilities::castDoubleToFloat(const double value) {
	return __double2float_rd(value);
}
