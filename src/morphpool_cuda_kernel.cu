#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

#include <vector>

#define THREADS_FORWARD 1024
#define THREADS_BACKWARD 64

#define EPS 1e-8

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
//#else
//__device__ double atomicAdd(double* a, double b) { return b; }
//#endif

namespace {

template <typename scalar_t>
__global__ void morphpool_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3> Input,
    const torch::PackedTensorAccessor32<scalar_t,3> Mask,
    torch::PackedTensorAccessor32<scalar_t,5> output,
    torch::PackedTensorAccessor32<scalar_t,6> output_idx,
    size_t batch_size,
    size_t input_channels,
    size_t height,
    size_t width,
    int num_morph,
    int kernel_size) {

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    const int mid = kernel_size / 2;
    const int n = index / height / width;
    const int c = n % input_channels;
    const int h = (index / width) % height;
    const int w = index % width;
    const int batch = n / input_channels;

    if (n < batch_size && h < height && w < width) {
        for (int m=0; m<num_morph; m++)
        {
            scalar_t max_val = 0.;
            int max_y = 0;
            int max_x = 0;
            bool first = true;

            for (int i=0; i<kernel_size; i++) {
                const int y = h + i - mid;
                if (y >= 0 && y < height) {
                    for (int j=0; j<kernel_size; j++) {
                        const int x = w + j - mid;
                        if (x >= 0 && x < width) {
                            if (Mask[m][i][j] == 1)
                            {
                                if (Input[n][y][x] * Mask[m][i][j] > max_val || first) {
                                    max_val = Input[n][y][x] * Mask[m][i][j];
                                    max_y = y;
                                    max_x = x;
                                    first = false;
                                }
                            }
                        }
                    }
                }
            }

            output[batch][c][m][h][w] = max_val;
            output_idx[batch][c][m][h][w][0] = max_y;
            output_idx[batch][c][m][h][w][1] = max_x;
        }
    }
}

template <typename scalar_t>
__global__ void morphpool_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5> Grad,
    const torch::PackedTensorAccessor32<scalar_t,4> Input,
    const torch::PackedTensorAccessor32<scalar_t,3> Mask,
    const torch::PackedTensorAccessor32<scalar_t,6> Indices,
    const torch::PackedTensorAccessor32<scalar_t,5> Output_fwd,
    torch::PackedTensorAccessor32<scalar_t,4> output,
    torch::PackedTensorAccessor32<scalar_t,3> output_mask,
    size_t batch_size,
    size_t input_channels,
    size_t height,
    size_t width,
    int num_morph,
    int kernel_size) {

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    scalar_t value = 0.;
    const int mid = kernel_size / 2;
    const int n = index / height / width;
    const int c = n % input_channels;
    const int h = (index / width) % height;
    const int w = index % width;
    const int batch = n / input_channels;


    if (n < batch_size && h < height && w < width) {
        for (int m=0; m<num_morph; m++)
        {
            for (int i=0; i<kernel_size; i++) {
                const int y = h + i - mid;
                if (y >= 0 && y < height) {
                    for (int j=0; j<kernel_size; j++) {
                        const int x = w + j - mid;
                        if (x >= 0 && x < width) {
                            const int i_mask =  kernel_size - i - 1;
                            const int j_mask =  kernel_size - j - 1;

                            const int max_y = Indices[batch][c][m][y][x][0];
                            const int max_x = Indices[batch][c][m][y][x][1];

                            if (w == max_x && h == max_y) {
                                if (Mask[m][i_mask][j_mask] == 1)
                                {
                                    value = value + Mask[m][i_mask][j_mask] * Grad[batch][c][m][y][x];
                                    gpuAtomicAdd(&output_mask[m][i_mask][j_mask], Input[batch][c][h][w] * Grad[batch][c][m][y][x]);
                                }
                            }
                        }
                    }
                }
            }
        }
        output[batch][c][h][w] = value;
    }
}



template <typename scalar_t>
__global__ void morphpool3d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4> Input,
    const torch::PackedTensorAccessor32<scalar_t,4> Mask,
    torch::PackedTensorAccessor32<scalar_t,6> output,
    torch::PackedTensorAccessor32<scalar_t,7> output_idx,
    size_t batch_size,
    size_t input_channels,
    size_t depth,
    size_t height,
    size_t width,
    int num_morph,
    int kernel_size) {

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    const int mid = kernel_size / 2;
    const int n = index / depth / height / width;
    const int c = n % input_channels;
    const int d = (index / height / width) % depth;
    const int h = (index / width) % height;
    const int w = index % width;
    const int batch = n / input_channels;

    if (n < batch_size && d < depth && h < height && w < width) {
        for (int m=0; m<num_morph; m++)
        {
            scalar_t max_val = 0.;
            int max_z = 0;
            int max_y = 0;
            int max_x = 0;
            bool first = true;

            for (int k=0; k<kernel_size; k++) {
                const int z = d + k - mid;
                if (z >=0 && z < depth) {
                    for (int i=0; i<kernel_size; i++) {
                        const int y = h + i - mid;
                        if (y >= 0 && y < height) {
                            for (int j=0; j<kernel_size; j++) {
                                const int x = w + j - mid;
                                if (x >= 0 && x < width) {
                                    if (Mask[m][k][i][j] == 1)
                                    {
                                        if (Input[n][z][y][x] * Mask[m][k][i][j] > max_val || first) {
                                            max_val = Input[n][z][y][x] * Mask[m][k][i][j];
                                            max_z = z;
                                            max_y = y;
                                            max_x = x;
                                            first = false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            output[batch][c][m][d][h][w] = max_val;
            output_idx[batch][c][m][d][h][w][0] = max_z;
            output_idx[batch][c][m][d][h][w][1] = max_y;
            output_idx[batch][c][m][d][h][w][2] = max_x;
        }
    }
}

template <typename scalar_t>
__global__ void morphpool3d_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6> Grad,
    const torch::PackedTensorAccessor32<scalar_t,5> Input,
    const torch::PackedTensorAccessor32<scalar_t,4> Mask,
    const torch::PackedTensorAccessor32<scalar_t,7> Indices,
    const torch::PackedTensorAccessor32<scalar_t,6> Output_fwd,
    torch::PackedTensorAccessor32<scalar_t,5> output,
    torch::PackedTensorAccessor32<scalar_t,4> output_mask,
    size_t batch_size,
    size_t input_channels,
    size_t depth,
    size_t height,
    size_t width,
    int num_morph,
    int kernel_size) {

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    scalar_t value = 0.;
    const int mid = kernel_size / 2;
    const int n = index / depth / height / width;
    const int c = n % input_channels;
    const int d = (index / height / width) % depth;
    const int h = (index / width) % height;
    const int w = index % width;
    const int batch = n / input_channels;


    if (n < batch_size && d < depth && h < height && w < width) {
        for (int m=0; m<num_morph; m++)
        {
            for (int k=0; k<kernel_size; k++) {
                const int z = d + k - mid;
                if (z >=0 && z < depth) {
                    for (int i=0; i<kernel_size; i++) {
                        const int y = h + i - mid;
                        if (y >= 0 && y < height) {
                            for (int j=0; j<kernel_size; j++) {
                                const int x = w + j - mid;
                                if (x >= 0 && x < width) {
                                    const int k_mask = kernel_size - k - 1;
                                    const int i_mask = kernel_size - i - 1;
                                    const int j_mask = kernel_size - j - 1;

                                    const int max_z = Indices[batch][c][m][z][y][x][0];
                                    const int max_y = Indices[batch][c][m][z][y][x][1];
                                    const int max_x = Indices[batch][c][m][z][y][x][2];

                                    if (w == max_x && h == max_y && d == max_z) {
                                        if (Mask[m][k_mask][i_mask][j_mask] == 1)
                                        {
                                            value = value + Mask[m][k_mask][i_mask][j_mask] * Grad[batch][c][m][z][y][x];
                                            gpuAtomicAdd(&output_mask[m][k_mask][i_mask][j_mask], Input[batch][c][d][h][w] * Grad[batch][c][m][z][y][x]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output[batch][c][d][h][w] = value;
    }
}

} // namespace

std::vector<torch::Tensor> morphpool_cuda_forward(
    torch::Tensor input,
    torch::Tensor mask,
    int num_morph,
    int kernel_size) {

    auto output = torch::zeros_like(input);
    auto output_idx = torch::zeros_like(input);

    if (mask.dim() == 3) {
        const auto batch = input.size(0);
        const auto channel = input.size(1);
        const auto height = input.size(2);
        const auto width = input.size(3);

        auto vInput = input.view({-1, height, width});
        const auto batch_size = vInput.size(0);

        output.resize_({batch, channel, num_morph, height, width});
        output_idx.resize_({batch, channel, num_morph, height, width, 2});
        output.fill_(0);
        output_idx.fill_(0);

        const int threads = THREADS_FORWARD;
        const dim3 blocks((height * width + threads - 1) / threads, batch_size);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        AT_ASSERT(input.numel() < std::numeric_limits<int32_t>::max());
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "morphpool_cuda_forward_cuda", ([&] {
            morphpool_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                vInput.packed_accessor32<scalar_t,3>(),
                mask.packed_accessor32<scalar_t,3>(),
                output.packed_accessor32<scalar_t,5>(),
                output_idx.packed_accessor32<scalar_t,6>(),
                batch_size,
                channel,
                height,
                width,
                num_morph,
                kernel_size);
        }));

    }
    else {
        if (mask.dim() == 4) {
            const auto batch = input.size(0);
            const auto channel = input.size(1);
            const auto depth = input.size(2);
            const auto height = input.size(3);
            const auto width = input.size(4);

            auto vInput = input.view({-1, depth, height, width});
            const auto batch_size = vInput.size(0);

            output.resize_({batch, channel, num_morph, depth, height, width});
            output_idx.resize_({batch, channel, num_morph, depth, height, width, 3});
            output.fill_(0);
            output_idx.fill_(0);

            const int threads = THREADS_FORWARD;
            const dim3 blocks((depth * height * width + threads - 1) / threads, batch_size);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            AT_ASSERT(input.numel() < std::numeric_limits<int32_t>::max());
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "morphpool3d_cuda_forward_cuda", ([&] {
                morphpool3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                    vInput.packed_accessor32<scalar_t,4>(),
                    mask.packed_accessor32<scalar_t,4>(),
                    output.packed_accessor32<scalar_t,6>(),
                    output_idx.packed_accessor32<scalar_t,7>(),
                    batch_size,
                    channel,
                    depth,
                    height,
                    width,
                    num_morph,
                    kernel_size);
            }));
        }
    }
    return {output, output_idx};
}

std::vector<torch::Tensor> morphpool_cuda_backward(
    torch::Tensor grad,
    torch::Tensor input,
    torch::Tensor mask,
    torch::Tensor input_indices,
    torch::Tensor output_fwd,
    int num_morph,
    int kernel_size) {

    auto output = torch::zeros_like(input);
    auto output_mask = torch::zeros_like(mask);

    if (mask.dim() == 3) {
        const auto batch = input.size(0);
        const auto channel = input.size(1);
        const auto height = input.size(2);
        const auto width = input.size(3);

        auto vInput = input.view({-1, height, width});
        const auto batch_size = vInput.size(0);

    //  ORIGINAL

        const int threads = THREADS_BACKWARD;
        const dim3 blocks((height * width + threads - 1) / threads, batch_size);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        AT_ASSERT(vInput.numel() < std::numeric_limits<int32_t>::max());
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(vInput.scalar_type(), "morphpool_backward_cuda", ([&] {
            morphpool_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad.packed_accessor32<scalar_t,5>(),
                input.packed_accessor32<scalar_t,4>(),
                mask.packed_accessor32<scalar_t,3>(),
                input_indices.packed_accessor32<scalar_t,6>(),
                output_fwd.packed_accessor32<scalar_t,5>(),
                output.packed_accessor32<scalar_t,4>(),
                output_mask.packed_accessor32<scalar_t,3>(),
                batch_size,
                channel,
                height,
                width,
                num_morph,
                kernel_size);
        }));

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
    else {
        if (mask.dim() == 4) {
            const auto batch = input.size(0);
            const auto channel = input.size(1);
            const auto depth = input.size(2);
            const auto height = input.size(3);
            const auto width = input.size(4);

            auto vInput = input.view({-1, depth, height, width});
            const auto batch_size = vInput.size(0);

        //  ORIGINAL

            const int threads = THREADS_BACKWARD;
            const dim3 blocks((depth * height * width + threads - 1) / threads, batch_size);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            AT_ASSERT(vInput.numel() < std::numeric_limits<int32_t>::max());
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(vInput.scalar_type(), "morphpool3d_backward_cuda", ([&] {
                morphpool3d_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                    grad.packed_accessor32<scalar_t,6>(),
                    input.packed_accessor32<scalar_t,5>(),
                    mask.packed_accessor32<scalar_t,4>(),
                    input_indices.packed_accessor32<scalar_t,7>(),
                    output_fwd.packed_accessor32<scalar_t,6>(),
                    output.packed_accessor32<scalar_t,5>(),
                    output_mask.packed_accessor32<scalar_t,4>(),
                    batch_size,
                    channel,
                    depth,
                    height,
                    width,
                    num_morph,
                    kernel_size);
            }));

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }
    }

    return {output, output_mask};
}