#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include "../../reduction_utils.cuh"

namespace vllm {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

#define FP8_E4M3_MAX std::numeric_limits<c10::Float8_e4m3fn>::max()

template <typename scalar_t>
__device__ __forceinline__ c10::Float8_e4m3fn scaled_fp8_conversion(
    const scalar_t val, const float inverted_scale) {
  float x = static_cast<float>(val) * inverted_scale;
  float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<c10::Float8_e4m3fn>(r);
}

// Compute the absolute maximum m of the input tensor and store
// m / float8_e4m3::max() in *scale. Each thread block performs a
// reduction tree and the memory in scale is atomically updated.
// So to get the right answer, *scale needs to be initialized to
// a value <= 0.0 and we need to wait for all thread blocks to
// finish before consuming *scale.
template <typename scalar_t>
__global__ void segmented_max_reduction(float* __restrict__ scale,
                                        const scalar_t* __restrict__ input,
                                        int64_t num_elems) {
  __shared__ float cache[1024];
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // First store maximum for all values processes by
  // the current thread in cache[threadIdx.x]
  scalar_t tmp = 0.0;
  while (i < num_elems) {
    float x = static_cast<float>(input[i]);
    tmp = max(tmp, fabs(x));
    i += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = tmp;

  __syncthreads();

  // Now perform parallel reduction within the thread block
  int ib = blockDim.x / 2;
  while (ib != 0) {
    if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x]) {
      cache[threadIdx.x] = cache[threadIdx.x + ib];
    }
    __syncthreads();
    ib /= 2;
  }
  // Finally, since cache[0] contains the maximum for this thread block,
  // atomically write the max to the target location
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale,
                   cache[0] / std::numeric_limits<c10::Float8_e4m3fn>::max());
  }
}

template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

typedef struct __align__(4) {
  c10::Float8_e4m3fn x;
  c10::Float8_e4m3fn y;
  c10::Float8_e4m3fn z;
  c10::Float8_e4m3fn w;
}
float8x4_t;

template <typename scalar_t>
__device__ float thread_max_vec(scalar_t const* __restrict__ input,
                                int64_t const num_elems, int const tid,
                                int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vectorized_in =
      reinterpret_cast<vec4_t<scalar_t> const*>(input);

  int const num_vec_elems = num_elems >> 2;
  float absmax_val = 0.0f;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    absmax_val = max(absmax_val, fabs(in_vec.x));
    absmax_val = max(absmax_val, fabs(in_vec.y));
    absmax_val = max(absmax_val, fabs(in_vec.z));
    absmax_val = max(absmax_val, fabs(in_vec.w));
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    absmax_val = max(absmax_val, fabs(input[i]));
  }

  return absmax_val;
}

template <typename scalar_t>
__device__ void scaled_fp8_conversion_vec(c10::Float8_e4m3fn* __restrict__ out,
                                          scalar_t const* __restrict__ input,
                                          float const inverted_scale,
                                          int64_t const num_elems,
                                          int const tid, int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vectorized_in =
      reinterpret_cast<vec4_t<scalar_t> const*>(input);
  float8x4_t* vectorized_out = reinterpret_cast<float8x4_t*>(out);

  int const num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    float8x4_t out_vec;

    out_vec.x = scaled_fp8_conversion(in_vec.x, inverted_scale);
    out_vec.y = scaled_fp8_conversion(in_vec.y, inverted_scale);
    out_vec.z = scaled_fp8_conversion(in_vec.z, inverted_scale);
    out_vec.w = scaled_fp8_conversion(in_vec.w, inverted_scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion(input[i], inverted_scale);
  }
}

template <typename scalar_t>
__global__ void scaled_fp8_quant_kernel(c10::Float8_e4m3fn* __restrict__ out,
                                        const scalar_t* __restrict__ input,
                                        const float* __restrict__ scale,
                                        int64_t num_elems) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / (*scale);

  scaled_fp8_conversion_vec(out, input, inverted_scale, num_elems, tid,
                            blockDim.x * gridDim.x);
}

template <typename scalar_t>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel(
    c10::Float8_e4m3fn* __restrict__ out, float* __restrict__ scale,
    scalar_t const* __restrict__ input, const int hidden_size) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;

  scalar_t const* __restrict__ token_input = &input[token_idx * hidden_size];
  c10::Float8_e4m3fn* __restrict__ token_output = &out[token_idx * hidden_size];

  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  float absmax_val = 0.0f;
  if (can_vectorize) {
    absmax_val = thread_max_vec(token_input, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float const x = static_cast<float>(token_input[i]);
      absmax_val = max(absmax_val, fabs(x));
    }
  }

  float const block_absmax_val_maybe = blockReduceMax(absmax_val);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val / FP8_E4M3_MAX;
  }
  __syncthreads();

  float const inverted_scale = FP8_E4M3_MAX / block_absmax_val;
  if (can_vectorize) {
    scaled_fp8_conversion_vec(token_output, token_input, inverted_scale,
                              hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      token_output[i] = scaled_fp8_conversion(token_input[i], inverted_scale);
    }
  }
}

}  // namespace vllm

void static_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                             torch::Tensor const& input,  // [..., d]
                             torch::Tensor const& scale)  // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel", [&] {
        vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<c10::Float8_e4m3fn>(), input.data_ptr<scalar_t>(),
            scale.data_ptr<float>(), num_elems);
      });
}

void dynamic_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                              torch::Tensor const& input,  // [..., d]
                              torch::Tensor& scale)        // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel", [&] {
        vllm::segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
            scale.data_ptr<float>(), input.data_ptr<scalar_t>(), num_elems);
        vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<c10::Float8_e4m3fn>(), input.data_ptr<scalar_t>(),
            scale.data_ptr<float>(), num_elems);
      });
}

void dynamic_per_token_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                                        torch::Tensor const& input,  // [..., d]
                                        torch::Tensor& scales) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_per_token_scaled_fp8_quant_kernel", [&] {
        vllm::dynamic_per_token_scaled_fp8_quant_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<c10::Float8_e4m3fn>(), scales.data_ptr<float>(),
                input.data_ptr<scalar_t>(), hidden_size);
      });
}