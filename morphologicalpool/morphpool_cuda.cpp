#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> morphpool_cuda_forward(
    at::Tensor input,
    at::Tensor mask,
    int num_morph,
    int kernel_size);

std::vector<at::Tensor> morphpool_cuda_backward(
    at::Tensor grad,
    at::Tensor input,
    at::Tensor mask,
    at::Tensor input_indices,
    at::Tensor output_fwd,
    int num_morph,
    int kernel_size);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> morphpool_forward(
    at::Tensor input,
    at::Tensor mask,
    int num_morph,
    int kernel_size) {
  CHECK_INPUT(input);
  CHECK_INPUT(mask);

  return morphpool_cuda_forward(input, mask, num_morph, kernel_size);
}

std::vector<at::Tensor> morphpool_backward(
    at::Tensor grad,
    at::Tensor input,
    at::Tensor mask,
    at::Tensor input_indices,
    at::Tensor output_fwd,
    int num_morph,
    int kernel_size) {
  CHECK_INPUT(grad);
  CHECK_INPUT(input);
  CHECK_INPUT(mask);
  CHECK_INPUT(input_indices);
  CHECK_INPUT(output_fwd);

  return morphpool_cuda_backward(
      grad,
      input,
      mask,
      input_indices,
      output_fwd,
      num_morph,
      kernel_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &morphpool_forward, "morphpool forward (CUDA)");
  m.def("backward", &morphpool_backward, "morphpool backward (CUDA)");
}