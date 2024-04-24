#include <torch/extension.h>

torch::Tensor forward(torch::Tensor Q_d, torch::Tensor K_d, torch::Tensor V_d);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}