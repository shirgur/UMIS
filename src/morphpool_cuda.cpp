#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> morphpool_cuda_forward(
    torch::Tensor input,
    torch::Tensor mask,
    int num_morph,
    int kernel_size);

std::vector<torch::Tensor> morphpool_cuda_backward(
    torch::Tensor grad,
    torch::Tensor input,
    torch::Tensor mask,
    torch::Tensor input_indices,
    torch::Tensor output_fwd,
    int num_morph,
    int kernel_size);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> morphpool_forward(
    torch::Tensor input,
    torch::Tensor mask,
    int num_morph,
    int kernel_size) {
  CHECK_INPUT(input);
  CHECK_INPUT(mask);

  return morphpool_cuda_forward(input, mask, num_morph, kernel_size);
}

std::vector<torch::Tensor> morphpool_backward(
    torch::Tensor grad,
    torch::Tensor input,
    torch::Tensor mask,
    torch::Tensor input_indices,
    torch::Tensor output_fwd,
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